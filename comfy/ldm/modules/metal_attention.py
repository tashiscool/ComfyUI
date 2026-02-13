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
- Tiled K/V loading into threadgroup shared memory
- Hybrid routing: SDPA for small sequences, Metal FA for large ones
"""

import torch
from torch.nn import functional as F
import math
import logging

logger = logging.getLogger(__name__)

_compiled_libs = {}

METAL_FLASH_ATTENTION_SIMD_SOURCE = '#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n\n// Tile dimensions\nconstant uint BQ = {bq};       // query rows per SIMD group (must be 8)\nconstant uint BK = {bk};       // KV block size\nconstant uint HD = {head_dim}; // head dimension\nconstant uint HD_TILES = HD / 8;  // number of 8-wide tiles in head_dim\nconstant uint NUM_SIMD = {num_simd};  // SIMD groups per threadgroup\nconstant uint BQ_TOTAL = BQ * NUM_SIMD;  // total query rows per threadgroup\n\nkernel void flash_attention_fwd(\n    device const half* Q [[buffer(0)]],\n    device const half* K [[buffer(1)]],\n    device const half* V [[buffer(2)]],\n    device half* O [[buffer(3)]],\n    constant uint& N_q [[buffer(4)]],       // original (unpadded) query length\n    constant uint& N_kv [[buffer(5)]],\n    constant uint& batch_heads [[buffer(6)]],\n    constant float& scale [[buffer(7)]],\n    constant uint& N_q_padded [[buffer(8)]], // padded query length (stride for Q/O)\n    uint2 tgid [[threadgroup_position_in_grid]],\n    uint tid [[thread_index_in_threadgroup]],\n    uint simd_lane [[thread_index_in_simdgroup]],\n    uint simd_idx [[simdgroup_index_in_threadgroup]]\n) {{\n    uint bh = tgid.y;\n    if (bh >= batch_heads) return;\n\n    // Each SIMD group handles its own BQ=8 query rows\n    uint q_start = tgid.x * BQ_TOTAL + simd_idx * BQ;\n\n    // Base pointers: Q and O use padded stride, K/V use N_kv stride\n    device const half* Q_bh = Q + (uint64_t)bh * N_q_padded * HD;\n    device const half* K_bh = K + (uint64_t)bh * N_kv * HD;\n    device const half* V_bh = V + (uint64_t)bh * N_kv * HD;\n    device half* O_bh = O + (uint64_t)bh * N_q_padded * HD;\n\n    // Shared memory for K/V tiles \xe2\x80\x94 shared across all SIMD groups\n    threadgroup half k_s[BK * HD];\n    threadgroup half v_s[BK * HD];\n\n    // Thread-to-element mapping for simdgroup_matrix 8x8 on Apple Silicon:\n    // Uses 4-quadrant layout (verified on M4 Pro, matches Apple MLX reference):\n    //   Lanes 0-7:   rows 0-3, cols 0-3 (top-left)\n    //   Lanes 8-15:  rows 0-3, cols 4-7 (top-right)\n    //   Lanes 16-23: rows 4-7, cols 0-3 (bottom-left)\n    //   Lanes 24-31: rows 4-7, cols 4-7 (bottom-right)\n    // Row reduction: xor(1) pairs within quadrant, xor(8) across quadrants\n    uint qid = simd_lane / 4;\n    uint my_row0 = (qid & 4u) + ((simd_lane / 2u) % 4u);\n    uint my_col0 = (qid & 2u) * 2u + (simd_lane % 2u) * 2u;\n\n    float row_max_val = -INFINITY;\n    float row_sum_val = 0.0f;\n\n    // Output accumulators: HD_TILES float8x8 matrices\n    simdgroup_float8x8 o_acc[{hd_tiles}];\n    for (uint t = 0; t < {hd_tiles}; t++) {{\n        o_acc[t] = simdgroup_float8x8(0.0f);\n    }}\n\n    // KEY OPTIMIZATION: Cache Q tiles in registers \xe2\x80\x94 loaded once, reused across all KV blocks\n    simdgroup_half8x8 q_cached[{hd_tiles}];\n    if (q_start + 8 <= N_q) {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            simdgroup_load(q_cached[hd_t], Q_bh + (uint64_t)q_start * HD + hd_t * 8, HD);\n        }}\n    }} else if (q_start < N_q) {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            simdgroup_load(q_cached[hd_t], Q_bh + (uint64_t)min(q_start, N_q - 1) * HD + hd_t * 8, HD);\n        }}\n    }} else {{\n        for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n            q_cached[hd_t] = simdgroup_half8x8(0);\n        }}\n    }}\n\n    uint threads_per_tg = {threads_per_tg};\n\n    // Iterate over KV blocks\n    for (uint kv_start = 0; kv_start < N_kv; kv_start += BK) {{\n        uint blen = min(BK, N_kv - kv_start);\n\n        // Cooperative K/V load using ALL threads across SIMD groups\n        uint total_elems = blen * HD;\n        for (uint idx = tid; idx < total_elems; idx += threads_per_tg) {{\n            uint r = idx / HD;\n            uint c = idx % HD;\n            uint64_t src = (uint64_t)(kv_start + r) * HD + c;\n            k_s[r * HD + c] = K_bh[src];\n            v_s[r * HD + c] = V_bh[src];\n        }}\n        for (uint idx = tid; idx < (BK - blen) * HD; idx += threads_per_tg) {{\n            uint r = blen + idx / HD;\n            uint c = idx % HD;\n            k_s[r * HD + c] = 0;\n            v_s[r * HD + c] = 0;\n        }}\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        uint bk_tiles = (blen + 7) / 8;\n        for (uint bk_t = 0; bk_t < bk_tiles; bk_t++) {{\n            // Compute S = Q_cached \xc3\x97 K^T using cached Q registers\n            simdgroup_float8x8 s_acc = simdgroup_float8x8(0.0f);\n            for (uint hd_t = 0; hd_t < {hd_tiles}; hd_t++) {{\n                simdgroup_half8x8 k_tile;\n                simdgroup_load(k_tile, k_s + bk_t * 8 * HD + hd_t * 8, HD, ulong2(0, 0), true);\n                simdgroup_multiply_accumulate(s_acc, q_cached[hd_t], k_tile, s_acc);\n            }}\n\n            float2 s_vals = float2(s_acc.thread_elements()[0], s_acc.thread_elements()[1]) * scale;\n\n            // Mask invalid KV positions\n            uint kv_col0 = kv_start + bk_t * 8 + my_col0;\n            if (kv_col0 >= N_kv) s_vals[0] = -INFINITY;\n            if (kv_col0 + 1 >= N_kv) s_vals[1] = -INFINITY;\n            if (q_start + my_row0 >= N_q) {{ s_vals[0] = -INFINITY; s_vals[1] = -INFINITY; }}\n\n            // Online softmax\n            float local_max = max(s_vals[0], s_vals[1]);\n            float row_max_new = local_max;\n            row_max_new = max(row_max_new, simd_shuffle_xor(row_max_new, 1));\n            row_max_new = max(row_max_new, simd_shuffle_xor(row_max_new, 8));\n            float prev_max = row_max_val;\n            float new_max = max(prev_max, row_max_new);\n            // Guard against NaN when both maxes are -INFINITY (all-masked padding rows)\n            float alpha = (new_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);\n\n            for (uint t = 0; t < {hd_tiles}; t++) {{\n                o_acc[t].thread_elements()[0] *= alpha;\n                o_acc[t].thread_elements()[1] *= alpha;\n            }}\n            row_sum_val *= alpha;\n            row_max_val = new_max;\n\n            float2 p_vals;\n            p_vals[0] = exp(s_vals[0] - new_max);\n            p_vals[1] = exp(s_vals[1] - new_max);\n            float local_sum = p_vals[0] + p_vals[1];\n            local_sum += simd_shuffle_xor(local_sum, 1);\n            local_sum += simd_shuffle_xor(local_sum, 8);\n            row_sum_val += local_sum;\n\n            // P \xc3\x97 V accumulation\n            simdgroup_half8x8 p_tile;\n            p_tile.thread_elements()[0] = half(p_vals[0]);\n            p_tile.thread_elements()[1] = half(p_vals[1]);\n            for (uint hd_t = 0; hd_t < HD_TILES; hd_t++) {{\n                simdgroup_half8x8 v_tile;\n                simdgroup_load(v_tile, v_s + bk_t * 8 * HD + hd_t * 8, HD);\n                simdgroup_multiply_accumulate(o_acc[hd_t], p_tile, v_tile, o_acc[hd_t]);\n            }}\n        }}\n\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }}\n\n    // Normalize by row_sum and write output\n    // IMPORTANT: simdgroup_store is cooperative \xe2\x80\x94 ALL threads must participate.\n    // Write unconditionally to padded output; invalid rows will be sliced off.\n    float inv_sum = (row_sum_val > 0.0f) ? (1.0f / row_sum_val) : 0.0f;\n    for (uint t = 0; t < {hd_tiles}; t++) {{\n        simdgroup_half8x8 out_tile;\n        out_tile.thread_elements()[0] = half(o_acc[t].thread_elements()[0] * inv_sum);\n        out_tile.thread_elements()[1] = half(o_acc[t].thread_elements()[1] * inv_sum);\n        simdgroup_store(out_tile, O_bh + (uint64_t)q_start * HD + t * 8, HD);\n    }}\n}}\n'


def _get_lib(head_dim, dtype):
    """Get or compile the Metal flash attention shader library."""
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
            f'Compiled Metal Flash Attention: head_dim={head_dim}'
            f', BQ={bq}'
            f', BK={bk}'
            f', SIMD_groups={num_simd}'
            f' ({threads_per_tg} threads/tg)')
    return _compiled_libs[key]


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


def metal_flash_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    """
    Flash Attention via Metal compute shaders on Apple Silicon.

    Uses a hybrid strategy:
    - Small sequences where SDPA fits: PyTorch SDPA (faster, Apple-optimized)
    - Large sequences where SDPA would OOM: Metal Flash Attention (O(N) memory)

    Falls back to SDPA when Metal FA is not applicable (non-MPS device,
    mask present, unsupported dtype/head_dim).
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

    # Tile parameters
    bq = 8
    num_simd = 4
    bq_total = bq * num_simd
    threads_per_tg = num_simd * 32
    pad_q = (bq_total - n_q % bq_total) % bq_total
    if pad_q > 0:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    n_q_padded = q_c.shape[1]

    out = torch.empty_like(q_c)

    lib = _get_lib(dim_head, q.dtype)
    scale = 1.0 / math.sqrt(dim_head)

    num_q_blocks = n_q_padded // bq_total

    lib.flash_attention_fwd(
        q_c, k_c, v_c, out,
        n_q, n_kv, BH, scale, n_q_padded,
        threads=(num_q_blocks * threads_per_tg, BH),
        group_size=(threads_per_tg, 1))

    # Strip padding
    if pad_q > 0:
        out = out[:, :n_q, :]

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
