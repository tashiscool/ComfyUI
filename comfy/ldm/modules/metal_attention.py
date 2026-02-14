"""
Metal Flash Attention for Apple Silicon (M-series) GPUs.

Implements the Flash Attention algorithm (Dao et al., 2022) as a Metal compute
shader using simdgroup_matrix hardware matmul units for Q*K^T and P*V.

Designed for ComfyUI's attention system, optimized for Wan 2.2 14B with
head_dim=128 and long sequences (~126K tokens for 81-frame 480p video).

Key properties:
- O(N) memory (vs O(N^2) for standard attention) - prevents OOM/kernel panics
- Hardware-accelerated simdgroup_matrix 8x8 multiply for dot products
- Float32 accumulation for numerical stability
- Hybrid routing: SDPA for small sequences, Metal FA for large ones

v3 kernel (optimized hybrid, default):
- BK=32, 4 SIMD groups / 128 threads
- K in threadgroup memory (transposed loads need coalescing)
- V loaded directly from device (non-transposed, naturally coalesced)
- Block-level online softmax with exp2 (single-cycle instruction)
- Vectorized half4 cooperative K loading
- 2.2 TFLOPS on M4 Pro (34% ALU utilization, 2.9x over v1)
"""

import torch
from torch.nn import functional as F
import math
import logging

logger = logging.getLogger(__name__)

# ============================================================
# v1 kernel (original, kept for benchmarking/fallback)
# ============================================================

_compiled_libs = {}

METAL_FLASH_ATTENTION_SIMD_SOURCE = '#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n\n// Tile dimensions\nconstant uint BQ = {bq};       // query rows per SIMD group (must be 8)\nconstant uint BK = {bk};       // KV block size\nconstant uint HD = {head_dim}; // head dimension\nconstant uint HD_TILES = HD / 8;  // number of 8-wide tiles in head_dim\nconstant uint NUM_SIMD = {num_simd};  // SIMD groups per threadgroup\nconstant uint BQ_TOTAL = BQ * NUM_SIMD;  // total query rows per threadgroup\n\nkernel void flash_attention_fwd(\n    device const half* Q [[buffer(0)]],\n    device const half* K [[buffer(1)]],\n    device const half* V [[buffer(2)]],\n    device half* O [[buffer(3)]],\n    constant uint& N_q [[buffer(4)]],       // original (unpadded) query length\n    constant uint& N_kv [[buffer(5)]],\n    constant uint& batch_heads [[buffer(6)]],\n    constant float& scale [[buffer(7)]],\n    constant uint& N_q_padded [[buffer(8)]], // padded query length (stride for Q/O)\n    uint2 tgid [[threadgroup_position_in_grid]],\n    uint tid [[thread_index_in_threadgroup]],\n    uint simd_lane [[thread_index_in_simdgroup]],\n    uint simd_idx [[simdgroup_index_in_threadgroup]]\n) {{\n    uint bh = tgid.y;\n    if (bh >= batch_heads) return;\n\n    // Each SIMD group handles its own BQ=8 query rows\n    uint q_start = tgid.x * BQ_TOTAL + simd_idx * BQ;\n\n    // Base pointers: Q and O use padded stride, K/V use N_kv stride\n    device const half* Q_bh = Q + (uint64_t)bh * N_q_padded * HD;\n    device const half* K_bh = K + (uint64_t)bh * N_kv * HD;\n    device const half* V_bh = V + (uint64_t)bh * N_kv * HD;\n    device half* O_bh = O + (uint64_t)bh * N_q_padded * HD;\n\n    // Shared memory for K/V tiles \xe2\x80\x94 shared across all SIMD groups\n    threadgroup half k_s[BK * HD];\n    threadgroup half v_s[BK * HD];\n\n    // Thread-to-element mapping for simdgroup_matrix 8x8 on Apple Silicon:\n    // Uses 4-quadrant layout (verified on M4 Pro, matches Apple MLX reference):\n    //   Lanes 0-7:   rows 0-3, cols 0-3 (top-left)\n    //   Lanes 8-15:  rows 0-3, cols 4-7 (top-right)\n    //   Lanes 16-23: rows 4-7, cols 0-3 (bottom-left)\n    //   Lanes 24-31: rows 4-7, cols 4-7 (bottom-right)\n    // Row reduction: xor(1) pairs within quadrant, xor(8) across quadrants\n    uint qid = simd_lane / 4;\n    uint my_row0 = (qid & 4u) + ((simd_lane / 2u) % 4u);\n    uint my_col0 = (qid & 2u) * 2u + (simd_lane % 2u) * 2u;\n\n    float row_max_val = -INFINITY;\n    float row_sum_val = 0.0f;\n\n    // Output accumulators: HD_TILES float8x8 matrices\n    simdgroup_float8x8 o_acc[{hd_tiles}];\n    for (uint t = 0; t < {hd_tiles}; t++) {{\n        o_acc[t] = simdgroup_float8x8(0.0f);\n    }}\n\n    // KEY OPTIMIZATION: Cache Q tiles in registers \xe2\x80\x94 loaded once, reused across all KV blocks\n    simdgroup_half8x8 q_cached[{hd_tiles}];\n    if (q_start + 8 <= N_q) {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            simdgroup_load(q_cached[hd_t], Q_bh + (uint64_t)q_start * HD + hd_t * 8, HD);\n        }}\n    }} else if (q_start < N_q) {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            simdgroup_load(q_cached[hd_t], Q_bh + (uint64_t)min(q_start, N_q - 1) * HD + hd_t * 8, HD);\n        }}\n    }} else {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            q_cached[hd_t] = simdgroup_half8x8(0);\n        }}\n    }}\n\n    uint threads_per_tg = {threads_per_tg};\n\n    // Iterate over KV blocks\n    for (uint kv_start = 0; kv_start < N_kv; kv_start += BK) {{\n        uint blen = min(BK, N_kv - kv_start);\n\n        // Cooperative K/V load using ALL threads across SIMD groups\n        uint total_elems = blen * HD;\n        for (uint idx = tid; idx < total_elems; idx += threads_per_tg) {{\n            uint r = idx / HD;\n            uint c = idx % HD;\n            uint64_t src = (uint64_t)(kv_start + r) * HD + c;\n            k_s[r * HD + c] = K_bh[src];\n            v_s[r * HD + c] = V_bh[src];\n        }}\n        for (uint idx = tid; idx < (BK - blen) * HD; idx += threads_per_tg) {{\n            uint r = blen + idx / HD;\n            uint c = idx % HD;\n            k_s[r * HD + c] = 0;\n            v_s[r * HD + c] = 0;\n        }}\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        uint bk_tiles = (blen + 7) / 8;\n        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {{\n            // Compute S = Q_cached \xc3\x97 K^T using cached Q registers\n            simdgroup_float8x8 s_acc = simdgroup_float8x8(0.0f);\n            for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n                simdgroup_half8x8 k_tile;\n                simdgroup_load(k_tile, k_s + bk_t * 8 * HD + hd_t * 8, HD, ulong2(0, 0), true);\n                simdgroup_multiply_accumulate(s_acc, q_cached[hd_t], k_tile, s_acc);\n            }}\n\n            float2 s_vals = float2(s_acc.thread_elements()[0], s_acc.thread_elements()[1]) * scale;\n\n            // Mask invalid KV positions\n            uint kv_col0 = kv_start + bk_t * 8 + my_col0;\n            if (kv_col0 >= N_kv) s_vals[0] = -INFINITY;\n            if (kv_col0 + 1 >= N_kv) s_vals[1] = -INFINITY;\n            if (q_start + my_row0 >= N_q) {{ s_vals[0] = -INFINITY; s_vals[1] = -INFINITY; }}\n\n            // Online softmax\n            float local_max = max(s_vals[0], s_vals[1]);\n            float row_max_new = local_max;\n            row_max_new = max(row_max_new, simd_shuffle_xor(row_max_new, 1));\n            row_max_new = max(row_max_new, simd_shuffle_xor(row_max_new, 8));\n            float prev_max = row_max_val;\n            float new_max = max(prev_max, row_max_new);\n            // Guard against NaN when both maxes are -INFINITY (all-masked padding rows)\n            float alpha = (new_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);\n\n            for (uint t = 0; t < {hd_tiles}; t++) {{\n                o_acc[t].thread_elements()[0] *= alpha;\n                o_acc[t].thread_elements()[1] *= alpha;\n            }}\n            row_sum_val *= alpha;\n            row_max_val = new_max;\n\n            float2 p_vals;\n            p_vals[0] = exp(s_vals[0] - new_max);\n            p_vals[1] = exp(s_vals[1] - new_max);\n            float local_sum = p_vals[0] + p_vals[1];\n            local_sum += simd_shuffle_xor(local_sum, 1);\n            local_sum += simd_shuffle_xor(local_sum, 8);\n            row_sum_val += local_sum;\n\n            // P \xc3\x97 V accumulation\n            simdgroup_half8x8 p_tile;\n            p_tile.thread_elements()[0] = half(p_vals[0]);\n            p_tile.thread_elements()[1] = half(p_vals[1]);\n            for (uint hd_t = 0; hd_t < HD_TILES; hd_t++) {{\n                simdgroup_half8x8 v_tile;\n                simdgroup_load(v_tile, v_s + bk_t * 8 * HD + hd_t * 8, HD);\n                simdgroup_multiply_accumulate(o_acc[hd_t], p_tile, v_tile, o_acc[hd_t]);\n            }}\n        }}\n\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }}\n\n    // Normalize by row_sum and write output\n    // IMPORTANT: simdgroup_store is cooperative \xe2\x80\x94 ALL threads must participate.\n    // Write unconditionally to padded output; invalid rows will be sliced off.\n    float inv_sum = (row_sum_val > 0.0f) ? (1.0f / row_sum_val) : 0.0f;\n    for (uint t = 0; t < {hd_tiles}; t++) {{\n        simdgroup_half8x8 out_tile;\n        out_tile.thread_elements()[0] = half(o_acc[t].thread_elements()[0] * inv_sum);\n        out_tile.thread_elements()[1] = half(o_acc[t].thread_elements()[1] * inv_sum);\n        simdgroup_store(out_tile, O_bh + (uint64_t)q_start * HD + t * 8, HD);\n    }}\n}}\n'


def _get_lib(head_dim, dtype):
    """Get or compile the Metal flash attention v1 shader library."""
    key = (head_dim, dtype)
    if key not in _compiled_libs:
        if head_dim % 8 != 0:
            raise ValueError(f'head_dim must be divisible by 8, got {head_dim}')
        bq = 8
        bk = 32
        num_simd = 4
        hd_tiles = head_dim // 8
        threads_per_tg = num_simd * 32
        source = METAL_FLASH_ATTENTION_SIMD_SOURCE.format(
            bq=bq, bk=bk, head_dim=head_dim,
            hd_tiles=hd_tiles, threads_per_tg=threads_per_tg,
            num_simd=num_simd)
        _compiled_libs[key] = torch.mps.compile_shader(source)
        logger.info(
            f'Compiled Metal Flash Attention v1: head_dim={head_dim}'
            f', BQ={bq}'
            f', BK={bk}'
            f', SIMD_groups={num_simd}'
            f' ({threads_per_tg} threads/tg)')
    return _compiled_libs[key]


# ============================================================
# v2 kernel (MFA-style: BK=128, 2 SIMD groups, device loads)
# ============================================================

_compiled_libs_v2 = {}

# Module-level version selector: 'v1', 'v2a', 'v2b', 'v3'
METAL_FA_VERSION = 'v3'


def _generate_v2_source(head_dim, cache_q):
    """Generate Metal Flash Attention v2 kernel source.

    Args:
        head_dim: Head dimension (must be divisible by 32).
        cache_q: If True, cache all Q tiles in registers (v2b).
                 If False, reload Q per head block of 32 (v2a, matches MFA for HD>96).
    """
    hd_tiles = head_dim // 8
    num_hb = head_dim // 32

    # Build Q initialization block
    if cache_q:
        q_init = (
            '    // v2b: Cache all Q tiles in registers\n'
            '    simdgroup_half8x8 q_cached[' + str(hd_tiles) + '];\n'
            '    if (q_start < N_q) {\n'
            '        for (uint t = 0; t < ' + str(hd_tiles) + '; t++)\n'
            '            simdgroup_load(q_cached[t], Q_bh + (uint64_t)q_start * HD + t * 8, HD);\n'
            '    } else {\n'
            '        for (uint t = 0; t < ' + str(hd_tiles) + '; t++)\n'
            '            q_cached[t] = simdgroup_half8x8(0);\n'
            '    }\n'
        )
    else:
        q_init = '    // v2a: Q reloaded per head block of 32\n'

    # Build QK computation block
    if cache_q:
        qk_code = (
            '        // S = Q_cached * K^T (Q in registers, direct K device loads)\n'
            '        for (uint bk_t = 0; bk_t < BK_TILES; bk_t++) {\n'
            '            for (uint hd_t = 0; hd_t < ' + str(hd_tiles) + '; hd_t++) {\n'
            '                simdgroup_half8x8 k_tile;\n'
            '                simdgroup_load(k_tile, K_bh + (uint64_t)(kv_start + bk_t * 8) * HD + hd_t * 8, HD, ulong2(0,0), true);\n'
            '                simdgroup_multiply_accumulate(s_acc[bk_t], q_cached[hd_t], k_tile, s_acc[bk_t]);\n'
            '            }\n'
            '        }\n'
        )
    else:
        qk_code = (
            '        // S = Q * K^T (Q reloaded per head block, direct K device loads)\n'
            '        for (uint hb = 0; hb < ' + str(num_hb) + '; hb++) {\n'
            '            uint d_offset = hb * 32;\n'
            '            simdgroup_half8x8 q_tiles[4];\n'
            '            if (q_start < N_q) {\n'
            '                for (uint d = 0; d < 4; d++)\n'
            '                    simdgroup_load(q_tiles[d], Q_bh + (uint64_t)q_start * HD + d_offset + d * 8, HD);\n'
            '            } else {\n'
            '                for (uint d = 0; d < 4; d++)\n'
            '                    q_tiles[d] = simdgroup_half8x8(0);\n'
            '            }\n'
            '            for (uint bk_t = 0; bk_t < BK_TILES; bk_t++) {\n'
            '                for (uint d = 0; d < 4; d++) {\n'
            '                    simdgroup_half8x8 k_tile;\n'
            '                    simdgroup_load(k_tile, K_bh + (uint64_t)(kv_start + bk_t * 8) * HD + d_offset + d * 8, HD, ulong2(0,0), true);\n'
            '                    simdgroup_multiply_accumulate(s_acc[bk_t], q_tiles[d], k_tile, s_acc[bk_t]);\n'
            '                }\n'
            '            }\n'
            '        }\n'
        )

    # Assemble full kernel
    HDT = str(hd_tiles)
    HD = str(head_dim)

    source = (
        '#include <metal_stdlib>\n'
        '#include <metal_simdgroup_matrix>\n'
        'using namespace metal;\n'
        '\n'
        'constant uint BQ = 8;\n'
        'constant uint BK = 128;\n'
        'constant uint HD = ' + HD + ';\n'
        'constant uint HD_TILES = ' + HDT + ';\n'
        'constant uint BK_TILES = 16;\n'
        'constant uint NUM_SIMD = 2;\n'
        'constant uint BQ_TOTAL = 16;\n'
        'constant float LOG2E_CONST = 1.442695041f;\n'
        '\n'
        'kernel void flash_attention_v2_fwd(\n'
        '    device const half* Q [[buffer(0)]],\n'
        '    device const half* K [[buffer(1)]],\n'
        '    device const half* V [[buffer(2)]],\n'
        '    device half* O [[buffer(3)]],\n'
        '    constant uint& N_q [[buffer(4)]],\n'
        '    constant uint& N_kv [[buffer(5)]],\n'
        '    constant uint& batch_heads [[buffer(6)]],\n'
        '    constant float& scale [[buffer(7)]],\n'
        '    constant uint& N_q_padded [[buffer(8)]],\n'
        '    constant uint& N_kv_padded [[buffer(9)]],\n'
        '    uint2 tgid [[threadgroup_position_in_grid]],\n'
        '    uint tid [[thread_index_in_threadgroup]],\n'
        '    uint simd_lane [[thread_index_in_simdgroup]],\n'
        '    uint simd_idx [[simdgroup_index_in_threadgroup]]\n'
        ') {\n'
        '    uint bh = tgid.y;\n'
        '    if (bh >= batch_heads) return;\n'
        '\n'
        '    uint q_start = tgid.x * BQ_TOTAL + simd_idx * BQ;\n'
        '\n'
        '    device const half* Q_bh = Q + (uint64_t)bh * N_q_padded * HD;\n'
        '    device const half* K_bh = K + (uint64_t)bh * N_kv_padded * HD;\n'
        '    device const half* V_bh = V + (uint64_t)bh * N_kv_padded * HD;\n'
        '    device half* O_bh = O + (uint64_t)bh * N_q_padded * HD;\n'
        '\n'
        '    // Morton-order thread-to-element mapping for 8x8 simdgroup_matrix\n'
        '    uint qid = simd_lane / 4;\n'
        '    uint my_row0 = (qid & 4u) + ((simd_lane / 2u) % 4u);\n'
        '    uint my_col0 = (qid & 2u) * 2u + (simd_lane % 2u) * 2u;\n'
        '\n'
        '    // Output accumulators (persistent across all KV blocks)\n'
        '    simdgroup_float8x8 o_acc[' + HDT + '];\n'
        '    for (uint t = 0; t < ' + HDT + '; t++) o_acc[t] = simdgroup_float8x8(0.0f);\n'
        '\n'
        '    float row_max = -INFINITY;\n'
        '    float row_sum = 0.0f;\n'
        '    float log2e_scale = LOG2E_CONST * scale;\n'
        '\n'
        + q_init +
        '\n'
        '    // === KV outer loop: BK=128 blocks, NO threadgroup memory, NO barriers ===\n'
        '    for (uint kv_start = 0; kv_start < N_kv_padded; kv_start += BK) {\n'
        '\n'
        '        // --- Phase 1: S = Q * K^T (16 float8x8 S tiles) ---\n'
        '        simdgroup_float8x8 s_acc[BK_TILES];\n'
        '        for (uint i = 0; i < BK_TILES; i++) s_acc[i] = simdgroup_float8x8(0.0f);\n'
        '\n'
        + qk_code +
        '\n'
        '        // --- Phase 2: Block-level online softmax ---\n'
        '        float block_max = -INFINITY;\n'
        '        for (uint bk_t = 0; bk_t < BK_TILES; bk_t++) {\n'
        '            float s0 = s_acc[bk_t].thread_elements()[0] * log2e_scale;\n'
        '            float s1 = s_acc[bk_t].thread_elements()[1] * log2e_scale;\n'
        '\n'
        '            // Mask invalid KV positions and padding query rows\n'
        '            uint kv_col0 = kv_start + bk_t * 8 + my_col0;\n'
        '            if (kv_col0 >= N_kv) s0 = -INFINITY;\n'
        '            if (kv_col0 + 1 >= N_kv) s1 = -INFINITY;\n'
        '            if (q_start + my_row0 >= N_q) { s0 = -INFINITY; s1 = -INFINITY; }\n'
        '\n'
        '            s_acc[bk_t].thread_elements()[0] = s0;\n'
        '            s_acc[bk_t].thread_elements()[1] = s1;\n'
        '            block_max = max(block_max, max(s0, s1));\n'
        '        }\n'
        '        // SIMD shuffle reduction (Morton order: xor 1 then xor 8)\n'
        '        block_max = max(block_max, simd_shuffle_xor(block_max, 1));\n'
        '        block_max = max(block_max, simd_shuffle_xor(block_max, 8));\n'
        '\n'
        '        // Online correction: rescale O accumulators\n'
        '        float new_max = max(row_max, block_max);\n'
        '        float alpha = (new_max == -INFINITY) ? 0.0f : fast::exp2(row_max - new_max);\n'
        '        for (uint t = 0; t < ' + HDT + '; t++) {\n'
        '            o_acc[t].thread_elements()[0] *= alpha;\n'
        '            o_acc[t].thread_elements()[1] *= alpha;\n'
        '        }\n'
        '        row_sum *= alpha;\n'
        '        row_max = new_max;\n'
        '\n'
        '        // --- Phase 3: Fused P computation + P*V accumulation ---\n'
        '        float block_sum = 0.0f;\n'
        '        for (uint bk_t = 0; bk_t < BK_TILES; bk_t++) {\n'
        '            float p0 = fast::exp2(s_acc[bk_t].thread_elements()[0] - new_max);\n'
        '            float p1 = fast::exp2(s_acc[bk_t].thread_elements()[1] - new_max);\n'
        '            block_sum += p0 + p1;\n'
        '\n'
        '            simdgroup_half8x8 p_tile;\n'
        '            p_tile.thread_elements()[0] = half(p0);\n'
        '            p_tile.thread_elements()[1] = half(p1);\n'
        '\n'
        '            for (uint hd_t = 0; hd_t < ' + HDT + '; hd_t++) {\n'
        '                simdgroup_half8x8 v_tile;\n'
        '                simdgroup_load(v_tile, V_bh + (uint64_t)(kv_start + bk_t * 8) * HD + hd_t * 8, HD);\n'
        '                simdgroup_multiply_accumulate(o_acc[hd_t], p_tile, v_tile, o_acc[hd_t]);\n'
        '            }\n'
        '        }\n'
        '        // SIMD shuffle reduction for sum\n'
        '        block_sum += simd_shuffle_xor(block_sum, 1);\n'
        '        block_sum += simd_shuffle_xor(block_sum, 8);\n'
        '        row_sum += block_sum;\n'
        '    }\n'
        '\n'
        '    // Normalize and write output\n'
        '    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;\n'
        '    for (uint t = 0; t < ' + HDT + '; t++) {\n'
        '        simdgroup_half8x8 out_tile;\n'
        '        out_tile.thread_elements()[0] = half(o_acc[t].thread_elements()[0] * inv_sum);\n'
        '        out_tile.thread_elements()[1] = half(o_acc[t].thread_elements()[1] * inv_sum);\n'
        '        simdgroup_store(out_tile, O_bh + (uint64_t)q_start * HD + t * 8, HD);\n'
        '    }\n'
        '}\n'
    )
    return source


def _get_lib_v2(head_dim, dtype, cache_q=False):
    """Get or compile the Metal flash attention v2 shader library."""
    key = (head_dim, dtype, cache_q)
    if key not in _compiled_libs_v2:
        if head_dim % 32 != 0:
            raise ValueError(f'v2 requires head_dim divisible by 32, got {head_dim}')
        source = _generate_v2_source(head_dim, cache_q)
        _compiled_libs_v2[key] = torch.mps.compile_shader(source)
        variant = 'v2b (Q cached)' if cache_q else 'v2a (Q reloaded/HD-block)'
        logger.info(
            f'Compiled Metal Flash Attention {variant}: HD={head_dim}'
            f', BK=128, 2 SIMD groups (64 threads/tg)')
    return _compiled_libs_v2[key]


# ============================================================
# Common helpers
# ============================================================

def _can_use_metal_fa(q, mask):
    """Check if Metal Flash Attention can handle this call."""
    if q.device.type != 'mps':
        return False
    if mask is not None:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def _fallback_sdpa(q, k, v, heads, b, dim_head, skip_reshape, skip_output_reshape):
    """Fallback to PyTorch SDPA."""
    if not skip_reshape:
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _estimate_sdpa_attn_bytes(heads, n_q, n_kv):
    """Estimate bytes for SDPA's attention matrix (float32 internal)."""
    return heads * n_q * n_kv * 4


_SDPA_MAX_BYTES = 2147483648


# ============================================================
# v3 kernel (hybrid: threadgroup staging + block softmax + exp2)
# ============================================================

_compiled_libs_v3 = {}


def _generate_v3_source(head_dim, bk=64, num_simd=4, half_s=False, device_v=False):
    """Generate Metal Flash Attention v3 kernel source.

    Hybrid architecture combining v1's threadgroup staging with v2's
    block-level softmax and exp2. Best of both worlds:
    - Cooperative K/V load into threadgroup (shared, no redundant bandwidth)
    - Block-level online softmax with exp2 (single-cycle instruction)
    - Q cached in registers
    - Configurable BK and SIMD groups

    Args:
        half_s: If True, accumulate S=Q*K^T in FP16 (lower register pressure,
                enables larger BK for fewer barrier overhead). MFA uses this
                on M4 for occupancy.
        device_v: If True, only K goes to threadgroup (V loaded from device).
                  Halves threadgroup memory, enabling BK up to 128 for HD=128.
                  V loads are non-transposed (coalesced) but redundant per SIMD.
    """
    hd_tiles = head_dim // 8
    bk_tiles = bk // 8
    bq_total = 8 * num_simd
    threads_per_tg = num_simd * 32
    HDT = str(hd_tiles)
    BKT = str(bk_tiles)
    HD = str(head_dim)
    BK = str(bk)
    NS = str(num_simd)
    BQT = str(bq_total)
    TPT = str(threads_per_tg)

    source = (
        '#include <metal_stdlib>\n'
        '#include <metal_simdgroup_matrix>\n'
        'using namespace metal;\n'
        '\n'
        'constant uint BQ = 8;\n'
        'constant uint BK = ' + BK + ';\n'
        'constant uint HD = ' + HD + ';\n'
        'constant uint HD_TILES = ' + HDT + ';\n'
        'constant uint BK_TILES = ' + BKT + ';\n'
        'constant uint NUM_SIMD = ' + NS + ';\n'
        'constant uint BQ_TOTAL = ' + BQT + ';\n'
        'constant uint THREADS_PER_TG = ' + TPT + ';\n'
        'constant float LOG2E_CONST = 1.442695041f;\n'
        '\n'
        'kernel void flash_attention_v3_fwd(\n'
        '    device const half* Q [[buffer(0)]],\n'
        '    device const half* K [[buffer(1)]],\n'
        '    device const half* V [[buffer(2)]],\n'
        '    device half* O [[buffer(3)]],\n'
        '    constant uint& N_q [[buffer(4)]],\n'
        '    constant uint& N_kv [[buffer(5)]],\n'
        '    constant uint& batch_heads [[buffer(6)]],\n'
        '    constant float& scale [[buffer(7)]],\n'
        '    constant uint& N_q_padded [[buffer(8)]],\n'
        '    uint2 tgid [[threadgroup_position_in_grid]],\n'
        '    uint tid [[thread_index_in_threadgroup]],\n'
        '    uint simd_lane [[thread_index_in_simdgroup]],\n'
        '    uint simd_idx [[simdgroup_index_in_threadgroup]]\n'
        ') {\n'
        '    uint bh = tgid.y;\n'
        '    if (bh >= batch_heads) return;\n'
        '\n'
        '    uint q_start = tgid.x * BQ_TOTAL + simd_idx * BQ;\n'
        '\n'
        '    device const half* Q_bh = Q + (uint64_t)bh * N_q_padded * HD;\n'
        '    device const half* K_bh = K + (uint64_t)bh * N_kv * HD;\n'
        '    device const half* V_bh = V + (uint64_t)bh * N_kv * HD;\n'
        '    device half* O_bh = O + (uint64_t)bh * N_q_padded * HD;\n'
        '\n'
        + ('    // Threadgroup memory for K only (V loaded from device)\n'
           '    threadgroup half k_s[BK * HD];\n'
           if device_v else
           '    // Threadgroup memory for K/V tiles (shared across all SIMD groups)\n'
           '    threadgroup half k_s[BK * HD];\n'
           '    threadgroup half v_s[BK * HD];\n'
          ) +
        '\n'
        '    // Morton-order thread-to-element mapping\n'
        '    uint qid = simd_lane / 4;\n'
        '    uint my_row0 = (qid & 4u) + ((simd_lane / 2u) % 4u);\n'
        '    uint my_col0 = (qid & 2u) * 2u + (simd_lane % 2u) * 2u;\n'
        '\n'
        '    // Output accumulators\n'
        '    simdgroup_float8x8 o_acc[' + HDT + '];\n'
        '    for (uint t = 0; t < ' + HDT + '; t++) o_acc[t] = simdgroup_float8x8(0.0f);\n'
        '\n'
        '    float row_max = -INFINITY;\n'
        '    float row_sum = 0.0f;\n'
        '    float log2e_scale = LOG2E_CONST * scale;\n'
        '\n'
        '    // Cache Q tiles in registers\n'
        '    simdgroup_half8x8 q_cached[' + HDT + '];\n'
        '    if (q_start + 8 <= N_q) {\n'
        '        for (uint t = 0; t < ' + HDT + '; t++)\n'
        '            simdgroup_load(q_cached[t], Q_bh + (uint64_t)q_start * HD + t * 8, HD);\n'
        '    } else if (q_start < N_q) {\n'
        '        for (uint t = 0; t < ' + HDT + '; t++)\n'
        '            simdgroup_load(q_cached[t], Q_bh + (uint64_t)min(q_start, N_q - 1) * HD + t * 8, HD);\n'
        '    } else {\n'
        '        for (uint t = 0; t < ' + HDT + '; t++)\n'
        '            q_cached[t] = simdgroup_half8x8(0);\n'
        '    }\n'
        '\n'
        '    // === KV outer loop: BK=' + BK + ' blocks, threadgroup staging, block softmax ===\n'
        '    for (uint kv_start = 0; kv_start < N_kv; kv_start += BK) {\n'
        '        uint blen = min(BK, N_kv - kv_start);\n'
        '\n'
        + (
           '        // Cooperative K-only load — vectorized half4 (V from device)\n'
           '        uint total_vec4 = (blen * HD) / 4;\n'
           '        device const half4* K4 = (device const half4*)(K_bh + (uint64_t)kv_start * HD);\n'
           '        threadgroup half4* k4 = (threadgroup half4*)k_s;\n'
           '        for (uint idx = tid; idx < total_vec4; idx += THREADS_PER_TG) {\n'
           '            k4[idx] = K4[idx];\n'
           '        }\n'
           '        // Zero-pad remaining K rows\n'
           '        uint pad_start = blen * HD;\n'
           '        uint pad_end = BK * HD;\n'
           '        for (uint idx = pad_start + tid; idx < pad_end; idx += THREADS_PER_TG) {\n'
           '            k_s[idx] = 0;\n'
           '        }\n'
           if device_v else
           '        // Cooperative K/V load — vectorized half4 (64-bit) loads\n'
           '        uint total_vec4 = (blen * HD) / 4;\n'
           '        device const half4* K4 = (device const half4*)(K_bh + (uint64_t)kv_start * HD);\n'
           '        device const half4* V4 = (device const half4*)(V_bh + (uint64_t)kv_start * HD);\n'
           '        threadgroup half4* k4 = (threadgroup half4*)k_s;\n'
           '        threadgroup half4* v4 = (threadgroup half4*)v_s;\n'
           '        for (uint idx = tid; idx < total_vec4; idx += THREADS_PER_TG) {\n'
           '            k4[idx] = K4[idx];\n'
           '            v4[idx] = V4[idx];\n'
           '        }\n'
           '        // Zero-pad remaining rows (scalar, only for partial blocks)\n'
           '        uint pad_start = blen * HD;\n'
           '        uint pad_end = BK * HD;\n'
           '        for (uint idx = pad_start + tid; idx < pad_end; idx += THREADS_PER_TG) {\n'
           '            k_s[idx] = 0;\n'
           '            v_s[idx] = 0;\n'
           '        }\n'
          ) +
        '        threadgroup_barrier(mem_flags::mem_threadgroup);\n'
        '\n'
        '        uint bk_tiles = (blen + 7) / 8;\n'
        '\n'
        '        // --- Phase 1: S = Q_cached * K^T (all BK_TILES S tiles) ---\n'
        + ('        simdgroup_half8x8 s_acc[' + BKT + '];\n'
           '        for (uint i = 0; i < ' + BKT + '; i++) s_acc[i] = simdgroup_half8x8(0);\n'
           if half_s else
           '        simdgroup_float8x8 s_acc[' + BKT + '];\n'
           '        for (uint i = 0; i < ' + BKT + '; i++) s_acc[i] = simdgroup_float8x8(0.0f);\n'
          ) +
        '\n'
        '        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {\n'
        '            for (uint hd_t = 0; hd_t < ' + HDT + '; hd_t++) {\n'
        '                simdgroup_half8x8 k_tile;\n'
        '                simdgroup_load(k_tile, k_s + bk_t * 8 * HD + hd_t * 8, HD, ulong2(0,0), true);\n'
        '                simdgroup_multiply_accumulate(s_acc[bk_t], q_cached[hd_t], k_tile, s_acc[bk_t]);\n'
        '            }\n'
        '        }\n'
        '\n'
        '        // --- Phase 2: Block-level online softmax with exp2 ---\n'
        '        float block_max = -INFINITY;\n'
        + ('        // FP16 S: read as float for softmax, store scaled back as half\n'
           '        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {\n'
           '            float s0 = float(s_acc[bk_t].thread_elements()[0]) * log2e_scale;\n'
           '            float s1 = float(s_acc[bk_t].thread_elements()[1]) * log2e_scale;\n'
           '\n'
           '            uint kv_col0 = kv_start + bk_t * 8 + my_col0;\n'
           '            if (kv_col0 >= N_kv) s0 = -INFINITY;\n'
           '            if (kv_col0 + 1 >= N_kv) s1 = -INFINITY;\n'
           '            if (q_start + my_row0 >= N_q) { s0 = -INFINITY; s1 = -INFINITY; }\n'
           '\n'
           '            s_acc[bk_t].thread_elements()[0] = half(s0);\n'
           '            s_acc[bk_t].thread_elements()[1] = half(s1);\n'
           '            block_max = max(block_max, max(s0, s1));\n'
           '        }\n'
           if half_s else
           '        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {\n'
           '            float s0 = s_acc[bk_t].thread_elements()[0] * log2e_scale;\n'
           '            float s1 = s_acc[bk_t].thread_elements()[1] * log2e_scale;\n'
           '\n'
           '            uint kv_col0 = kv_start + bk_t * 8 + my_col0;\n'
           '            if (kv_col0 >= N_kv) s0 = -INFINITY;\n'
           '            if (kv_col0 + 1 >= N_kv) s1 = -INFINITY;\n'
           '            if (q_start + my_row0 >= N_q) { s0 = -INFINITY; s1 = -INFINITY; }\n'
           '\n'
           '            s_acc[bk_t].thread_elements()[0] = s0;\n'
           '            s_acc[bk_t].thread_elements()[1] = s1;\n'
           '            block_max = max(block_max, max(s0, s1));\n'
           '        }\n'
          ) +
        '        block_max = max(block_max, simd_shuffle_xor(block_max, 1));\n'
        '        block_max = max(block_max, simd_shuffle_xor(block_max, 8));\n'
        '\n'
        '        float new_max = max(row_max, block_max);\n'
        '        float alpha = (new_max == -INFINITY) ? 0.0f : fast::exp2(row_max - new_max);\n'
        '        for (uint t = 0; t < ' + HDT + '; t++) {\n'
        '            o_acc[t].thread_elements()[0] *= alpha;\n'
        '            o_acc[t].thread_elements()[1] *= alpha;\n'
        '        }\n'
        '        row_sum *= alpha;\n'
        '        row_max = new_max;\n'
        '\n'
        '        // --- Phase 3: Fused P computation + P*V accumulation ---\n'
        '        float block_sum = 0.0f;\n'
        '        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {\n'
        + ('            float p0 = fast::exp2(float(s_acc[bk_t].thread_elements()[0]) - new_max);\n'
           '            float p1 = fast::exp2(float(s_acc[bk_t].thread_elements()[1]) - new_max);\n'
           if half_s else
           '            float p0 = fast::exp2(s_acc[bk_t].thread_elements()[0] - new_max);\n'
           '            float p1 = fast::exp2(s_acc[bk_t].thread_elements()[1] - new_max);\n'
          ) +
        '            block_sum += p0 + p1;\n'
        '\n'
        '            simdgroup_half8x8 p_tile;\n'
        '            p_tile.thread_elements()[0] = half(p0);\n'
        '            p_tile.thread_elements()[1] = half(p1);\n'
        '\n'
        '            for (uint hd_t = 0; hd_t < ' + HDT + '; hd_t++) {\n'
        '                simdgroup_half8x8 v_tile;\n'
        + ('                simdgroup_load(v_tile, V_bh + (uint64_t)(kv_start + bk_t * 8) * HD + hd_t * 8, HD);\n'
           if device_v else
           '                simdgroup_load(v_tile, v_s + bk_t * 8 * HD + hd_t * 8, HD);\n'
          ) +
        '                simdgroup_multiply_accumulate(o_acc[hd_t], p_tile, v_tile, o_acc[hd_t]);\n'
        '            }\n'
        '        }\n'
        '        block_sum += simd_shuffle_xor(block_sum, 1);\n'
        '        block_sum += simd_shuffle_xor(block_sum, 8);\n'
        '        row_sum += block_sum;\n'
        '\n'
        '        threadgroup_barrier(mem_flags::mem_threadgroup);\n'
        '    }\n'
        '\n'
        '    // Normalize and write output\n'
        '    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;\n'
        '    for (uint t = 0; t < ' + HDT + '; t++) {\n'
        '        simdgroup_half8x8 out_tile;\n'
        '        out_tile.thread_elements()[0] = half(o_acc[t].thread_elements()[0] * inv_sum);\n'
        '        out_tile.thread_elements()[1] = half(o_acc[t].thread_elements()[1] * inv_sum);\n'
        '        simdgroup_store(out_tile, O_bh + (uint64_t)q_start * HD + t * 8, HD);\n'
        '    }\n'
        '}\n'
    )
    return source


def _get_lib_v3(head_dim, dtype, bk=64, num_simd=4, half_s=False, device_v=False):
    """Get or compile the Metal flash attention v3 shader library."""
    key = (head_dim, dtype, bk, num_simd, half_s, device_v)
    if key not in _compiled_libs_v3:
        if head_dim % 8 != 0:
            raise ValueError(f'head_dim must be divisible by 8, got {head_dim}')
        source = _generate_v3_source(head_dim, bk, num_simd, half_s, device_v)
        _compiled_libs_v3[key] = torch.mps.compile_shader(source)
        threads = num_simd * 32
        s_type = 'FP16' if half_s else 'FP32'
        v_src = 'device' if device_v else 'threadgroup'
        logger.info(
            f'Compiled Metal Flash Attention v3: HD={head_dim}'
            f', BK={bk}, {num_simd} SIMD groups ({threads} threads/tg)'
            f', S_acc={s_type}, V={v_src}')
    return _compiled_libs_v3[key]


# ============================================================
# Dispatch: v1, v2, and v3 internal dispatch functions
# ============================================================

def _dispatch_v1(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, dtype):
    """Dispatch Metal Flash Attention v1 kernel."""
    bq = 8
    num_simd = 4
    bq_total = bq * num_simd
    threads_per_tg = num_simd * 32
    pad_q = (bq_total - n_q % bq_total) % bq_total
    if pad_q > 0:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    n_q_padded = q_c.shape[1]

    out = torch.empty_like(q_c)
    lib = _get_lib(dim_head, dtype)
    scale = 1.0 / math.sqrt(dim_head)
    num_q_blocks = n_q_padded // bq_total

    lib.flash_attention_fwd(
        q_c, k_c, v_c, out,
        n_q, n_kv, BH, scale, n_q_padded,
        threads=(num_q_blocks * threads_per_tg, BH),
        group_size=(threads_per_tg, 1))

    if pad_q > 0:
        out = out[:, :n_q, :]
    return out


def _dispatch_v2(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, dtype, cache_q=False):
    """Dispatch Metal Flash Attention v2 kernel."""
    bq_total = 16  # BQ=8 * NUM_SIMD=2
    threads_per_tg = 64  # 2 SIMD groups * 32

    # Pad Q to multiples of BQ_TOTAL=16
    pad_q = (bq_total - n_q % bq_total) % bq_total
    if pad_q > 0:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    n_q_padded = q_c.shape[1]

    # Pad K/V to multiples of BK=128 (prevents OOB device loads)
    pad_kv = (128 - n_kv % 128) % 128
    if pad_kv > 0:
        k_c = F.pad(k_c, (0, 0, 0, pad_kv))
        v_c = F.pad(v_c, (0, 0, 0, pad_kv))
    n_kv_padded = k_c.shape[1]

    out = torch.empty_like(q_c)
    lib = _get_lib_v2(dim_head, dtype, cache_q)
    scale = 1.0 / math.sqrt(dim_head)
    num_q_blocks = n_q_padded // bq_total

    lib.flash_attention_v2_fwd(
        q_c, k_c, v_c, out,
        n_q, n_kv, BH, scale, n_q_padded, n_kv_padded,
        threads=(num_q_blocks * threads_per_tg, BH),
        group_size=(threads_per_tg, 1))

    if pad_q > 0:
        out = out[:, :n_q, :]
    return out


def _dispatch_v3(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, dtype, bk=32, num_simd=4, half_s=False, device_v=True):
    """Dispatch Metal Flash Attention v3 kernel (hybrid)."""
    bq_total = 8 * num_simd
    threads_per_tg = num_simd * 32

    pad_q = (bq_total - n_q % bq_total) % bq_total
    if pad_q > 0:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    n_q_padded = q_c.shape[1]

    out = torch.empty_like(q_c)
    lib = _get_lib_v3(dim_head, dtype, bk, num_simd, half_s, device_v)
    scale = 1.0 / math.sqrt(dim_head)
    num_q_blocks = n_q_padded // bq_total

    lib.flash_attention_v3_fwd(
        q_c, k_c, v_c, out,
        n_q, n_kv, BH, scale, n_q_padded,
        threads=(num_q_blocks * threads_per_tg, BH),
        group_size=(threads_per_tg, 1))

    if pad_q > 0:
        out = out[:, :n_q, :]
    return out


# ============================================================
# Public API
# ============================================================

def metal_flash_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    """
    Flash Attention via Metal compute shaders on Apple Silicon.

    Uses a hybrid strategy:
    - Small sequences where SDPA fits: PyTorch SDPA (faster, Apple-optimized)
    - Large sequences where SDPA would OOM: Metal Flash Attention (O(N) memory)

    Falls back to SDPA when Metal FA is not applicable (non-MPS device,
    mask present, unsupported dtype/head_dim).

    Kernel version controlled by METAL_FA_VERSION module variable:
    - 'v1': Original kernel (BK=32, 4 SIMD groups, full threadgroup staging)
    - 'v2a': MFA-style (BK=128, 2 SIMD groups, Q reloaded per head block)
    - 'v2b': MFA-style with Q cached in registers
    - 'v3': Optimized hybrid (BK=32, 4 SIMD groups, K threadgroup + V device,
             block softmax + exp2, half4 loads) — 2.2 TFLOPS on M4 Pro
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if not _can_use_metal_fa(q, mask) or dim_head % 8 != 0 or dim_head > 256:
        return _fallback_sdpa(q, k, v, heads, b, dim_head, skip_reshape, skip_output_reshape)

    n_q = q.shape[2] if skip_reshape else q.shape[1]
    n_kv = k.shape[2] if skip_reshape else k.shape[1]
    if _estimate_sdpa_attn_bytes(heads, n_q, n_kv) < _SDPA_MAX_BYTES:
        return _fallback_sdpa(q, k, v, heads, b, dim_head, skip_reshape, skip_output_reshape)

    # Reshape to (B*H, N, dim_head) for the Metal kernel
    if skip_reshape:
        BH = b * heads
        q_c = q.reshape(BH, n_q, dim_head).contiguous()
        k_c = k.reshape(BH, n_kv, dim_head).contiguous()
        v_c = v.reshape(BH, n_kv, dim_head).contiguous()
    else:
        BH = b * heads
        q_c = q.view(b, n_q, heads, dim_head).permute(0, 2, 1, 3).contiguous().reshape(BH, n_q, dim_head)
        k_c = k.view(b, n_kv, heads, dim_head).permute(0, 2, 1, 3).contiguous().reshape(BH, n_kv, dim_head)
        v_c = v.view(b, n_kv, heads, dim_head).permute(0, 2, 1, 3).contiguous().reshape(BH, n_kv, dim_head)

    # Select kernel version
    ver = METAL_FA_VERSION
    if ver == 'v3':
        out = _dispatch_v3(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, q.dtype)
    elif ver.startswith('v2') and dim_head % 32 == 0:
        cache_q = (ver == 'v2b')
        out = _dispatch_v2(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, q.dtype, cache_q)
    else:
        out = _dispatch_v1(q_c, k_c, v_c, n_q, n_kv, BH, dim_head, q.dtype)

    if skip_output_reshape:
        return out.view(b, heads, n_q, dim_head)

    return out.view(b, heads, n_q, dim_head).permute(0, 2, 1, 3).contiguous().reshape(b, n_q, heads * dim_head)


METAL_FLASH_ATTENTION_AVAILABLE = False
try:
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'compile_shader'):
        if torch.backends.mps.is_available():
            METAL_FLASH_ATTENTION_AVAILABLE = True
except Exception:
    pass
