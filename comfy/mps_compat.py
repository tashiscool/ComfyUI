"""
MPS (Apple Silicon) compatibility patches for ComfyUI.

Consolidates all MPS-specific workarounds into a single module that
monkey-patches at startup. Only activates when the device is MPS.

Patches applied:
  A. Memory reporting — use mach_task_info phys_footprint for accurate memory
  B. torch.compile interception — prevent dynamo recompilation storms on MPS
  C. Model unload — skip partial unloading on unified memory (zero-copy full reload is faster)
  D. Unified memory unloading — free CPU-resident models when MPS needs space
"""

import ctypes
import gc
import json
import logging
import os
import platform
import sys
import time
import torch
import psutil

log = logging.getLogger(__name__)

_patches_applied = False


def _is_mps_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _is_truthy_env(name):
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_bool_or_none(name):
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == '':
        return None
    return raw.lower() in {'1', 'true', 'yes', 'on'}


def _env_float(name, default, minimum=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except Exception:
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _parse_csv_ints(raw, default):
    values = []
    for token in (raw or '').split(','):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception:
            continue
    return values if values else list(default)


def _macos_version_tuple():
    try:
        version = platform.mac_ver()[0]
        if not version:
            return (0, 0, 0)
        parts = [int(x) for x in version.split('.')]
        while len(parts) < 3:
            parts.append(0)
        return (parts[0], parts[1], parts[2])
    except Exception:
        return (0, 0, 0)


def _is_mfa_supported_platform():
    return _is_mps_available() and hasattr(torch, 'mps') and hasattr(torch.mps, 'compile_shader')


def _is_mfa_causal_mask_supported_platform():
    if not _is_mfa_supported_platform():
        return False
    # Draw Things gate: MFA causal mask support on macOS 13.4+.
    major, minor, _ = _macos_version_tuple()
    return (major, minor) >= (13, 4)


def _should_arm_mfa_guard():
    if _is_truthy_env('COMFY_MPS_FORCE_MFA_GUARD'):
        return True
    if _is_truthy_env('COMFY_MPS_SKIP_MFA_GUARD'):
        return False
    # Draw Things behavior: no guard for trusted causal-mask-capable path.
    return _is_mfa_supported_platform() and not _is_mfa_causal_mask_supported_platform()


_MFA_GUARD_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user', 'mps_mfa_guard.json')
_mfa_guard_locked = False

# Runtime pressure and generation admission controls.
_GIB = float(1024 ** 3)

# Tier 1: low available + moderate swap growth (acute pressure).
_SWAP_DELTA_ABORT_GB = _env_float('COMFY_MPS_SWAP_DELTA_ABORT_GB', 4.0, 0.0)
_AVAIL_ABORT_GB = _env_float('COMFY_MPS_AVAIL_ABORT_GB', 3.0, 0.0)
_SUSTAIN_SECONDS = _env_float('COMFY_MPS_SWAP_SUSTAIN_SECONDS', 10.0, 1.0)

# Tier 2: extreme swap growth regardless of "available RAM".
_SWAP_EXTREME_GB = _env_float('COMFY_MPS_SWAP_EXTREME_GB', 20.0, 0.0)
_SUSTAIN_EXTREME_S = _env_float('COMFY_MPS_SWAP_EXTREME_SUSTAIN_SECONDS', 60.0, 1.0)

_CHECK_INTERVAL = _env_float('COMFY_MPS_SWAP_CHECK_INTERVAL', 3.0, 0.5)

# Preflight generation admission.
_PREFLIGHT_MIN_AVAIL_GB = _env_float('COMFY_MPS_PREFLIGHT_MIN_AVAIL_GB', 4.0, 0.0)
_PREFLIGHT_MAX_SWAP_GB = _env_float('COMFY_MPS_PREFLIGHT_MAX_SWAP_GB', 32.0, 0.0)
_PREFLIGHT_MAX_FOOTPRINT_RATIO = _env_float('COMFY_MPS_PREFLIGHT_MAX_FOOTPRINT_RATIO', 0.98, 0.1)

# Pre-load cleanup (Patch D extension).
_PRELOAD_SWAP_CLEANUP_GB = _env_float('COMFY_MPS_PRELOAD_SWAP_CLEANUP_GB', 5.0, 0.0)
_PRELOAD_CLEANUP_COOLDOWN_S = _env_float('COMFY_MPS_PRELOAD_CLEANUP_COOLDOWN_S', 60.0, 1.0)

# Pressure lifecycle state.
_swap_baseline_bytes = 0
_pressure_since = 0.0
_active_generation = False
_abort_fired = False
_last_watchdog_check = 0.0
_last_preload_cleanup = 0.0
_runtime_pressure_state = {
    'active_generation': False,
    'under_pressure': False,
    'tier': 'none',
    'swap_delta_gb': 0.0,
    'swap_used_gb': 0.0,
    'available_gb': 0.0,
    'timestamp': 0.0,
}


def _read_mfa_guard_state():
    try:
        with open(_MFA_GUARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_mfa_guard_state(state):
    try:
        os.makedirs(os.path.dirname(_MFA_GUARD_FILE), exist_ok=True)
        tmp = _MFA_GUARD_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        os.replace(tmp, _MFA_GUARD_FILE)
    except Exception as e:
        log.debug('[MPS] MFA guard write failed: %s', e)


def _mark_mfa_guard_inflight(active, reason=''):
    state = _read_mfa_guard_state()
    state['inflight'] = bool(active)
    state['timestamp'] = int(time.time())
    if reason:
        state['reason'] = reason
    _write_mfa_guard_state(state)


def get_mps_pressure_state():
    """Expose current MPS pressure state for routing decisions in attention.py."""
    return dict(_runtime_pressure_state)


def _set_pressure_state(active_generation, under_pressure, tier, swap_delta_gb, swap_used_gb, available_gb):
    _runtime_pressure_state['active_generation'] = bool(active_generation)
    _runtime_pressure_state['under_pressure'] = bool(under_pressure)
    _runtime_pressure_state['tier'] = str(tier)
    _runtime_pressure_state['swap_delta_gb'] = float(swap_delta_gb)
    _runtime_pressure_state['swap_used_gb'] = float(swap_used_gb)
    _runtime_pressure_state['available_gb'] = float(available_gb)
    _runtime_pressure_state['timestamp'] = float(time.monotonic())


def _set_mfa_runtime_enabled(enabled, reason):
    enabled = bool(enabled)
    try:
        import comfy.ldm.modules.metal_attention as metal_attention
        if hasattr(metal_attention, 'set_metal_flash_attention_enabled'):
            metal_attention.set_metal_flash_attention_enabled(enabled, reason=reason)
        else:
            metal_attention.METAL_FLASH_ATTENTION_ENABLED = enabled
    except Exception as e:
        log.debug('[MPS] Could not set metal_attention runtime toggle: %s', e)

    try:
        from comfy.cli_args import args
        args.use_metal_flash_attention = enabled
    except Exception:
        pass

    def _rebind_imported_attention_aliases(attn_mod):
        """Rebind by-value imports of optimized_attention to current runtime backend."""
        rebound = 0
        target = getattr(attn_mod, 'optimized_attention', None)
        target_masked = getattr(attn_mod, 'optimized_attention_masked', target)
        if target is None:
            return

        for mod in list(sys.modules.values()):
            if mod is None or mod is attn_mod:
                continue
            mod_dict = getattr(mod, '__dict__', None)
            if not isinstance(mod_dict, dict):
                continue

            fn = mod_dict.get('optimized_attention')
            if callable(fn) and getattr(fn, '__module__', None) == attn_mod.__name__ and fn is not target:
                mod_dict['optimized_attention'] = target
                rebound += 1

            fn_masked = mod_dict.get('optimized_attention_masked')
            if callable(fn_masked) and getattr(fn_masked, '__module__', None) == attn_mod.__name__ and fn_masked is not target_masked:
                mod_dict['optimized_attention_masked'] = target_masked
                rebound += 1

        if rebound:
            log.info('[MPS] Rebound %d imported attention aliases to %s',
                     rebound, getattr(target, '__name__', 'optimized_attention'))

    try:
        import comfy.ldm.modules.attention as attn_mod
        if not enabled and hasattr(attn_mod, 'attention_sub_quad'):
            attn_mod.optimized_attention = attn_mod.attention_sub_quad
            attn_mod.optimized_attention_masked = attn_mod.attention_sub_quad
            attn_mod.MPS_FLASH_ATTENTION_IS_AVAILABLE = False
            log.warning('[MPS] Runtime route forced to sub_quad (MFA disabled)')
            _rebind_imported_attention_aliases(attn_mod)
        elif enabled:
            backend_available = bool(getattr(attn_mod, '_MFA_PACKAGE_AVAILABLE', False))
            try:
                import comfy.ldm.modules.metal_attention as metal_attention
                backend_available = backend_available or bool(getattr(metal_attention, 'METAL_FLASH_ATTENTION_AVAILABLE', False))
            except Exception:
                pass
            attn_mod.MPS_FLASH_ATTENTION_IS_AVAILABLE = backend_available
            if backend_available and hasattr(attn_mod, 'attention_metal_flash'):
                attn_mod.optimized_attention = attn_mod.attention_metal_flash
                attn_mod.optimized_attention_masked = attn_mod.attention_metal_flash
                _rebind_imported_attention_aliases(attn_mod)
            elif hasattr(attn_mod, 'attention_sub_quad'):
                attn_mod.optimized_attention = attn_mod.attention_sub_quad
                attn_mod.optimized_attention_masked = attn_mod.attention_sub_quad
                _rebind_imported_attention_aliases(attn_mod)
    except Exception:
        pass


def _apply_mfa_guardrails():
    """Draw Things-style MFA guard persistence and runtime policy."""
    global _mfa_guard_locked

    if _is_truthy_env('COMFY_MPS_RESET_MFA_GUARD'):
        _mark_mfa_guard_inflight(False, 'manual_reset')
        log.info('[MPS] MFA guard reset via COMFY_MPS_RESET_MFA_GUARD')

    if _is_truthy_env('COMFY_MPS_MFA_FORCE_OFF'):
        _mfa_guard_locked = True
        _set_mfa_runtime_enabled(False, 'forced off by COMFY_MPS_MFA_FORCE_OFF')
        return 1

    if _is_truthy_env('COMFY_MPS_MFA_FORCE_ON'):
        _mfa_guard_locked = False
        _set_mfa_runtime_enabled(True, 'forced on by COMFY_MPS_MFA_FORCE_ON')
        _mark_mfa_guard_inflight(False, 'force_on_clear')
        return 1

    # Draw Things equivalent of use_mfa_v3 user preference.
    user_pref = _env_bool_or_none('COMFY_MPS_USE_MFA_V3')
    if user_pref is False:
        _mfa_guard_locked = True
        _set_mfa_runtime_enabled(False, 'disabled by COMFY_MPS_USE_MFA_V3=0')
        return 1
    elif user_pref is True:
        _mfa_guard_locked = False

    state = _read_mfa_guard_state()
    if state.get('inflight') and not _is_truthy_env('COMFY_MPS_IGNORE_MFA_GUARD'):
        _mfa_guard_locked = True
        _set_mfa_runtime_enabled(False, 'previous run ended with MFA guard still armed (possible crash)')
        log.error('[MPS] MFA guard tripped from previous run; keeping MFA disabled. '
                  'Set COMFY_MPS_RESET_MFA_GUARD=1 to clear or COMFY_MPS_IGNORE_MFA_GUARD=1 to bypass once.')
        return 1

    return 0


# ---------------------------------------------------------------------------
# macOS mach_task_info for accurate physical footprint
# ---------------------------------------------------------------------------
# On Apple Silicon unified memory, neither torch.mps.current_allocated_memory()
# (misses CPU-side tensors) nor psutil.Process().memory_info().rss (misses
# Metal/GPU allocations) gives accurate total memory usage.
#
# mach_task_info(TASK_VM_INFO).phys_footprint is what Activity Monitor reports
# as "Memory" and correctly includes both CPU and GPU on unified memory.
# ---------------------------------------------------------------------------

class _TaskVMInfo(ctypes.Structure):
    """Minimal task_vm_info_data_t up to phys_footprint (macOS/ARM64)."""
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('region_count', ctypes.c_int32),
        ('page_size', ctypes.c_int32),
        ('resident_size', ctypes.c_uint64),
        ('resident_size_peak', ctypes.c_uint64),
        ('device', ctypes.c_uint64),
        ('device_peak', ctypes.c_uint64),
        ('internal', ctypes.c_uint64),
        ('internal_peak', ctypes.c_uint64),
        ('external', ctypes.c_uint64),
        ('external_peak', ctypes.c_uint64),
        ('reusable', ctypes.c_uint64),
        ('reusable_peak', ctypes.c_uint64),
        ('purgeable_volatile_pmap', ctypes.c_uint64),
        ('purgeable_volatile_resident', ctypes.c_uint64),
        ('purgeable_volatile_virtual', ctypes.c_uint64),
        ('compressed', ctypes.c_uint64),
        ('compressed_peak', ctypes.c_uint64),
        ('compressed_lifetime', ctypes.c_uint64),
        ('phys_footprint', ctypes.c_uint64),
    ]

_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = ctypes.sizeof(_TaskVMInfo) // ctypes.sizeof(ctypes.c_uint32)

try:
    _libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
except OSError:
    _libc = None


def _get_phys_footprint():
    """Get the process physical memory footprint via mach_task_info.

    Returns the same value Activity Monitor shows as 'Memory'.
    Includes CPU allocations, GPU/Metal allocations, and compressed pages.
    Returns None if the syscall fails.
    """
    if _libc is None:
        return None
    try:
        info = _TaskVMInfo()
        count = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
        ret = _libc.task_info(
            _libc.mach_task_self(),
            _TASK_VM_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if ret == 0:
            return info.phys_footprint
    except Exception:
        pass
    return None


def _patch_memory_reporting():
    """Patch A: Total memory cap + free memory = system available (no under-report).

    Total memory: we cap at effective_max (0.95 * recommended_max) so ComfyUI
    doesn't assume 48GB is all usable for MPS.

    Free memory: we use min(psutil.virtual_memory().available, effective_max).
    Stock ComfyUI already uses psutil.available for MPS; we only cap it. We
    previously used effective_max - footprint, which under-reported free (e.g.
    9.5 GB when the system had 22 GB), causing slice_attention to use 2–4x
    more steps and turning a 44 min run into 2+ hours. Using actual system
    available restores speed while still capping at our safe budget.
    """
    import comfy.model_management as mm

    recommended_max = torch.mps.recommended_max_memory()
    # 48 GB unified should behave like "at least 24 GB VRAM" — workflows that run on
    # 24 GB NVIDIA run here. We must not report a budget LOWER than 24 GB, or
    # get_free_memory() stays too small after Patch D and we never have "enough" free.
    # Use ~0.95 of recommended_max so we report a usable budget (e.g. 35–37 GB on
    # 48 GB Mac); Patch D unloads CPU models, then we have room for 14B + activations.
    # 40 GB budget: leaves ~8 GB for macOS + apps. At 44 GB the system beachballs
    # on every app switch during Wan 2.2 14B generation (peak 43.8 GB observed).
    # 40 GB is enough for 14B FP8 + activations; may trigger more model offloading
    # but prevents the OS from thrashing.
    effective_max = int(40 * (1024**3))

    # Verify phys_footprint works at startup
    test_footprint = _get_phys_footprint()
    if test_footprint is not None:
        log.info("[MPS] recommended_max: {:.1f} GB, effective budget: {:.1f} GB, "
                 "current footprint: {:.1f} GB (via mach_task_info)".format(
                     recommended_max / (1024**3),
                     effective_max / (1024**3),
                     test_footprint / (1024**3)))
    else:
        log.warning("[MPS] mach_task_info failed — falling back to driver_allocated_memory")
        log.info("[MPS] recommended_max: {:.1f} GB, effective budget: {:.1f} GB".format(
            recommended_max / (1024**3), effective_max / (1024**3)))

    _orig_get_total_memory = mm.get_total_memory

    def patched_get_total_memory(dev=None, torch_total_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        if hasattr(dev, 'type') and dev.type == 'mps':
            mem_total = effective_max
            mem_total_torch = mem_total
            if torch_total_too:
                return (mem_total, mem_total_torch)
            return mem_total
        return _orig_get_total_memory(dev, torch_total_too)

    mm.get_total_memory = patched_get_total_memory

    _orig_get_free_memory = mm.get_free_memory

    def patched_get_free_memory(dev=None, torch_free_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        if hasattr(dev, 'type') and dev.type == 'mps':
            # Use actual system RAM available so we don't under-report. Before this
            # fix we used effective_max - footprint (e.g. 9.5 GB), which made
            # slice_attention use 2–4x more steps → 44 min became 2+ hours. Stock
            # ComfyUI uses psutil.virtual_memory().available for MPS. Cap at
            # effective_max so we never claim more than our safe budget.
            system_available = psutil.virtual_memory().available
            mem_free = min(system_available, effective_max)
            if torch_free_too:
                return (mem_free, mem_free)
            return mem_free
        return _orig_get_free_memory(dev, torch_free_too)

    mm.get_free_memory = patched_get_free_memory

    # Update the module-level total_vram that was already computed with the old function
    mm.total_vram = patched_get_total_memory(mm.get_torch_device()) / (1024 * 1024)
    log.info("[MPS] Corrected total VRAM to {:.0f} MB".format(mm.total_vram))

    return 2  # number of patches applied


def _patch_torch_compile():
    """Patch B: Prevent torch.compile/dynamo from activating on MPS.

    sub_quad attention uses variable-size chunks, causing 64+ dynamo recompiles
    that exhaust memory. Custom nodes (TorchCompileModelWanVideoV2) can bypass
    TORCHDYNAMO_DISABLE=1 by setting torch._dynamo.config.cache_size_limit = 64.

    We intercept at three levels:
    1. torch.compile itself — return model unmodified on MPS
    2. set_torch_compile_wrapper — no-op on MPS
    3. torch._dynamo.config — reset cache_size_limit to 0
    """
    patch_count = 0
    _compile_skip_count = [0]  # mutable container for closure

    # 1. Wrap torch.compile to be a no-op on MPS
    _orig_torch_compile = torch.compile

    def patched_torch_compile(model=None, *args, **kwargs):
        if model is not None and hasattr(model, 'parameters'):
            try:
                device = next(model.parameters()).device
                if device.type == 'mps':
                    _compile_skip_count[0] += 1
                    if _compile_skip_count[0] <= 1:
                        log.info("[MPS] torch.compile skipped — not supported on MPS backend")
                    return model
            except StopIteration:
                pass
        # Also check if default device is MPS
        if _is_mps_available():
            try:
                dev = torch.device('mps')
                if torch.mps.is_available():
                    _compile_skip_count[0] += 1
                    if _compile_skip_count[0] <= 1:
                        log.info("[MPS] torch.compile skipped — MPS is the active backend "
                                 "(further skips will be silent)")
                    return model if model is not None else lambda m: m
            except Exception:
                pass
        return _orig_torch_compile(model, *args, **kwargs)

    torch.compile = patched_torch_compile
    patch_count += 1

    # 2. Patch set_torch_compile_wrapper to no-op on MPS
    try:
        import comfy_api.torch_helpers.torch_compile as tc
        _orig_set_wrapper = tc.set_torch_compile_wrapper

        def patched_set_torch_compile_wrapper(model, *args, **kwargs):
            log.info("[MPS] set_torch_compile_wrapper skipped — torch.compile disabled on MPS")
            return

        tc.set_torch_compile_wrapper = patched_set_torch_compile_wrapper
        patch_count += 1
    except ImportError:
        log.debug("[MPS] comfy_api.torch_helpers.torch_compile not found, skipping patch")

    # 3. Reset dynamo cache_size_limit to prevent custom nodes from re-enabling
    try:
        import torch._dynamo.config as dynamo_config
        dynamo_config.cache_size_limit = 0
        patch_count += 1
    except (ImportError, AttributeError):
        log.debug("[MPS] torch._dynamo.config not available, skipping cache_size_limit reset")

    return patch_count


def _patch_model_unload():
    """Patch C: Skip partial model unloading on MPS unified memory.

    On unified memory (Apple Silicon), partial unloading creates per-layer
    lowvram patches with synchronous CPU<->MPS copies. Full unload + reload
    is faster because CPU<->MPS is essentially zero-copy (same physical memory).
    """
    import comfy.model_management as mm

    _OrigLoadedModel = mm.LoadedModel

    _orig_model_unload = _OrigLoadedModel.model_unload

    def patched_model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None and hasattr(self.device, 'type') and self.device.type == 'mps':
            # On MPS unified memory, skip partial unload — go straight to full unload.
            # CPU<->MPS is zero-copy so full reload has no transfer cost.
            log.debug("[MPS] Skipping partial unload, using full unload (zero-copy unified memory)")
            self.model.detach(unpatch_weights)
            self.model_finalizer.detach()
            self.model_finalizer = None
            self.real_model = None
            return True
        return _orig_model_unload(self, memory_to_free, unpatch_weights)

    _OrigLoadedModel.model_unload = patched_model_unload
    return 1


def _patch_unified_memory_unloading():
    """Patch D: Free CPU-resident models when MPS needs space on unified memory.

    On Apple Silicon, CPU and MPS share the same physical memory pool. ComfyUI's
    free_memory() only considers models on the *same* device (line 617:
    ``if shift_model.device == device``). When loading a 13.6GB MPS model, it only
    finds other MPS-resident models to unload — the 10.8GB CPU text encoder is
    invisible, even though it consumes the same physical RAM.

    This patch wraps free_memory() to also consider CPU-resident models when the
    target device is MPS and the initial unloading pass didn't free enough.
    """
    import comfy.model_management as mm

    _orig_free_memory = mm.free_memory

    def patched_free_memory(memory_required, device, keep_loaded=[], for_dynamic=False, ram_required=0):
        global _last_preload_cleanup
        original_requested_gb = memory_required / (1024**3) if memory_required else 0
        total_budget = None

        # On MPS, ComfyUI often asks for activation memory (e.g. 68 GB for WAN 14B) on top of
        # model size. Cap to effective budget so we only unload until we have headroom;
        # 48 GB unified with 0.95 * recommended_max gives ~35 GB budget so Patch D can free enough.
        if hasattr(device, 'type') and device.type == 'mps':
            now = time.monotonic()
            if now - _last_preload_cleanup >= _PRELOAD_CLEANUP_COOLDOWN_S:
                try:
                    swap_gb = psutil.swap_memory().used / _GIB
                except Exception:
                    swap_gb = None
                if swap_gb is not None and swap_gb > _PRELOAD_SWAP_CLEANUP_GB:
                    log.warning(
                        "[MPS] *** PRE-LOAD: %.1f GB swap already used - clearing caches (cooldown %.0fs)",
                        swap_gb, _PRELOAD_CLEANUP_COOLDOWN_S)
                    gc.collect()
                    if hasattr(torch.mps, 'empty_cache') and callable(getattr(torch.mps, 'empty_cache', None)):
                        torch.mps.empty_cache()
                    gc.collect()
                    _last_preload_cleanup = now

            total_budget = mm.get_total_memory(device)
            total_budget_gb = total_budget / (1024**3)

            if memory_required > total_budget:
                log.warning(
                    "[MPS] *** IMPOSSIBLE REQUEST: ComfyUI asked to free %.1f GB on a machine with "
                    "unified memory budget ~%.1f GB. That can never be satisfied. Without capping, "
                    "the loader would still run, total usage (model + text encoder + activations) would "
                    "exceed RAM, and the OS would swap heavily → THRASHING. Capping request to %.1f GB.",
                    original_requested_gb, total_budget_gb, total_budget_gb
                )
                log.info("[MPS] Capping free_memory request {:.1f} GB -> {:.1f} GB (effective budget)".format(
                    original_requested_gb, total_budget_gb))
                memory_required = total_budget

            footprint_before = _get_phys_footprint()
            loaded = [(m.model.model.__class__.__name__, m.device.type, m.model_memory() / (1024**3))
                      for m in mm.current_loaded_models if not m.is_dead()]
            on_cpu = [x for x in loaded if x[1] == 'cpu']
            on_mps = [x for x in loaded if x[1] == 'mps']
            log.info(
                "[MPS] free_memory called: need {:.1f} GB, process footprint {:.1f} GB, "
                "loaded models: {} (MPS: {}, CPU: {})".format(
                    memory_required / (1024**3), (footprint_before or 0) / (1024**3),
                    loaded, len(on_mps), len(on_cpu)
                )
            )
            if on_cpu and total_budget:
                log.warning(
                    "[MPS] *** SAME-DEVICE ONLY: ComfyUI's built-in free_memory() only unloads models on "
                    "the *same* device (MPS). The %s CPU-resident model(s) above (e.g. text encoder ~10 GB) "
                    "use the same physical RAM but are INVISIBLE to that pass. Patch D will unload them "
                    "after the first pass if MPS still needs more.",
                    len(on_cpu)
                )

        # First run the original (handles same-device unloading only)
        result = _orig_free_memory(memory_required, device, keep_loaded, for_dynamic, ram_required)

        # On MPS unified memory, also consider unloading CPU-resident models
        if not (hasattr(device, 'type') and device.type == 'mps'):
            return result

        mem_free = mm.get_free_memory(device)
        if mem_free >= memory_required:
            log.info("[MPS] free_memory: sufficient after same-device pass ({:.1f} GB free)".format(
                mem_free / (1024**3)))
            return result

        # Still not enough — same-device pass couldn't satisfy. Unload CPU-resident models (Patch D).
        log.warning(
            "[MPS] *** SAME-DEVICE PASS INSUFFICIENT: After unloading MPS models we have %.1f GB free, "
            "need %.1f GB. Unloading CPU-resident models now (unified memory) — e.g. text encoder.",
            mem_free / (1024**3), memory_required / (1024**3)
        )
        cpu_device = torch.device('cpu')
        can_unload_cpu = []
        for i in range(len(mm.current_loaded_models) - 1, -1, -1):
            shift_model = mm.current_loaded_models[i]
            if shift_model.device == cpu_device:
                if shift_model not in keep_loaded and not shift_model.is_dead():
                    can_unload_cpu.append((shift_model.model_memory(), i))

        unloaded_indices = []
        for mem_size, i in sorted(can_unload_cpu, reverse=True):
            mem_free = mm.get_free_memory(device)
            if mem_free >= memory_required:
                break
            model_name = mm.current_loaded_models[i].model.model.__class__.__name__
            log.info("[MPS] Unloading CPU-resident {} ({:.1f} GB) to free unified memory".format(
                model_name, mem_size / (1024**3)))
            if mm.current_loaded_models[i].model_unload():
                unloaded_indices.append(i)

        for i in sorted(unloaded_indices, reverse=True):
            result.append(mm.current_loaded_models.pop(i))

        if unloaded_indices:
            gc.collect()
            mm.soft_empty_cache()
            if hasattr(torch.mps, 'empty_cache') and callable(getattr(torch.mps, 'empty_cache', None)):
                torch.mps.empty_cache()
            gc.collect()
            new_free = mm.get_free_memory(device)
            log.info("[MPS] After unified memory cleanup: {:.1f} GB free (needed {:.1f} GB)".format(
                new_free / (1024**3), memory_required / (1024**3)))

        # Warn only when genuinely short; we cap memory_required to total_budget, so we often
        # "need" 35 GB but only have 22 GB — that's still enough for a 13.6 GB model. Warn only
        # when free is below a safe threshold (e.g. 15 GB) so we don't alarm when headroom exists.
        final_free = mm.get_free_memory(device)
        final_free_gb = final_free / (1024**3)
        if final_free < memory_required:
            if final_free_gb < 15.0:
                log.warning(
                    "[MPS] *** THRASHING RISK: After all unloading we have %.1f GB free but need %.1f GB. "
                    "ComfyUI will load anyway; heavy swap likely. Close other apps or reduce resolution/frames.",
                    final_free_gb, memory_required / (1024**3)
                )
            else:
                log.info(
                    "[MPS] After unloading: %.1f GB free (capped request was %.1f GB). Headroom sufficient for load.",
                    final_free_gb, memory_required / (1024**3)
                )
        if total_budget and original_requested_gb > (total_budget / (1024**3)):
            log.info(
                "[MPS] (Original request was %.1f GB; capped to %.1f GB to avoid impossible "
                "target and reduce thrashing risk.)",
                original_requested_gb, total_budget / (1024**3)
            )

        return result

    mm.free_memory = patched_free_memory
    return 1


def _run_generation_preflight():
    """Draw Things-style generation admission check before sampling begins."""
    try:
        vm = psutil.virtual_memory()
        swap_used_gb = psutil.swap_memory().used / _GIB
        available_gb = vm.available / _GIB
    except Exception as e:
        log.debug("[MPS] Preflight memory probe failed: %s", e)
        return

    footprint = _get_phys_footprint()
    footprint_gb = (footprint / _GIB) if footprint is not None else None
    budget_gb = None
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if hasattr(dev, 'type') and dev.type == 'mps':
            budget_gb = mm.get_total_memory(dev) / _GIB
    except Exception:
        pass

    reasons = []
    if available_gb < _PREFLIGHT_MIN_AVAIL_GB and swap_used_gb > _PREFLIGHT_MAX_SWAP_GB:
        reasons.append(
            f"available {available_gb:.1f} GB < {_PREFLIGHT_MIN_AVAIL_GB:.1f} GB and "
            f"swap {swap_used_gb:.1f} GB > {_PREFLIGHT_MAX_SWAP_GB:.1f} GB")

    if budget_gb is not None and footprint_gb is not None:
        if footprint_gb > (budget_gb * _PREFLIGHT_MAX_FOOTPRINT_RATIO) and available_gb < (_PREFLIGHT_MIN_AVAIL_GB + 1.0):
            reasons.append(
                f"footprint {footprint_gb:.1f} GB is above "
                f"{_PREFLIGHT_MAX_FOOTPRINT_RATIO * 100:.0f}% of budget {budget_gb:.1f} GB")

    if reasons:
        message = "; ".join(reasons)
        log.error("[MPS] *** PREFLIGHT DENY: %s. Aborting before long swap thrash.", message)
        import comfy.model_management as mm
        mm.interrupt_current_processing(True)
        raise mm.InterruptProcessingException()

    if footprint_gb is None:
        log.info("[MPS] Preflight admit: avail %.1f GB, swap %.1f GB", available_gb, swap_used_gb)
    else:
        log.info(
            "[MPS] Preflight admit: avail %.1f GB, swap %.1f GB, footprint %.1f GB",
            available_gb, swap_used_gb, footprint_gb)


def _swap_watchdog_tick():
    """Abort sustained MPS swap thrash during generation."""
    global _last_watchdog_check, _pressure_since, _abort_fired
    if not _active_generation or _abort_fired:
        return

    now = time.monotonic()
    if now - _last_watchdog_check < _CHECK_INTERVAL:
        return
    _last_watchdog_check = now

    try:
        vm = psutil.virtual_memory()
        swap_used = psutil.swap_memory().used
    except Exception as e:
        log.debug("[MPS] Watchdog probe failed: %s", e)
        return

    swap_delta_gb = max(0.0, (swap_used - _swap_baseline_bytes) / _GIB)
    swap_used_gb = swap_used / _GIB
    avail_gb = vm.available / _GIB

    tier1 = swap_delta_gb > _SWAP_DELTA_ABORT_GB and avail_gb < _AVAIL_ABORT_GB
    tier2 = swap_delta_gb > _SWAP_EXTREME_GB
    under_pressure = tier1 or tier2

    if tier1:
        tier_label = "ACUTE"
        sustain_threshold = _SUSTAIN_SECONDS
    elif tier2:
        tier_label = "EXTREME_SWAP"
        sustain_threshold = _SUSTAIN_EXTREME_S
    else:
        tier_label = "none"
        sustain_threshold = 0.0

    _set_pressure_state(
        active_generation=_active_generation,
        under_pressure=under_pressure,
        tier=tier_label,
        swap_delta_gb=swap_delta_gb,
        swap_used_gb=swap_used_gb,
        available_gb=avail_gb,
    )

    if under_pressure:
        if _pressure_since == 0.0:
            _pressure_since = now
            log.warning(
                "[MPS] *** MEMORY PRESSURE (%s): swap +%.1f GB, %.1f GB available - monitoring",
                tier_label, swap_delta_gb, avail_gb)
        elif now - _pressure_since >= sustain_threshold:
            log.error(
                "[MPS] *** SWAP WATCHDOG (%s): swap +%.1f GB, %.1f GB available for %ds - aborting",
                tier_label, swap_delta_gb, avail_gb, int(now - _pressure_since))
            _abort_fired = True
            import comfy.model_management as mm
            mm.interrupt_current_processing(True)
    else:
        if _pressure_since != 0.0:
            log.info(
                "[MPS] Memory pressure resolved (swap +%.1f GB, %.1f GB available)",
                swap_delta_gb, avail_gb)
        _pressure_since = 0.0


def _patch_generation_lifecycle():
    """Track generation boundaries, baseline swap, and run preflight admission."""
    try:
        import comfy.samplers as samplers
        _orig_outer = samplers.CFGGuider.outer_sample

        def patched_outer(self, *args, **kwargs):
            global _swap_baseline_bytes, _pressure_since, _active_generation, _abort_fired, _last_watchdog_check
            try:
                _swap_baseline_bytes = psutil.swap_memory().used
            except Exception:
                _swap_baseline_bytes = 0

            _pressure_since = 0.0
            _abort_fired = False
            _last_watchdog_check = 0.0
            _active_generation = True
            _set_pressure_state(True, False, 'none', 0.0, _swap_baseline_bytes / _GIB, 0.0)
            log.debug("[MPS] Swap baseline: %.1f GB", _swap_baseline_bytes / _GIB)

            try:
                _run_generation_preflight()
                return _orig_outer(self, *args, **kwargs)
            finally:
                _active_generation = False
                _pressure_since = 0.0
                _abort_fired = False
                _set_pressure_state(False, False, 'none', 0.0, 0.0, 0.0)

        samplers.CFGGuider.outer_sample = patched_outer
        log.info(
            "[MPS] Generation lifecycle patch armed (preflight: avail>=%.1f GB or swap<=%.1f GB)",
            _PREFLIGHT_MIN_AVAIL_GB, _PREFLIGHT_MAX_SWAP_GB)
        return 1
    except (ImportError, AttributeError) as e:
        log.debug("[MPS] Could not patch CFGGuider.outer_sample for generation lifecycle: %s", e)
        return 0


def _patch_swap_watchdog():
    """Attach the swap watchdog to comfy.ops.run_every_op()."""
    try:
        import comfy.ops as ops
        _orig = ops.run_every_op

        def patched_run_every_op():
            _swap_watchdog_tick()
            _orig()

        ops.run_every_op = patched_run_every_op
        log.info(
            "[MPS] Swap watchdog armed - tier1: +%.0f GB swap & <%.0f GB avail for %ds, "
            "tier2: +%.0f GB swap for %ds (check every %.1fs)",
            _SWAP_DELTA_ABORT_GB, _AVAIL_ABORT_GB, int(_SUSTAIN_SECONDS),
            _SWAP_EXTREME_GB, int(_SUSTAIN_EXTREME_S), _CHECK_INTERVAL)
        return 1
    except (ImportError, AttributeError) as e:
        log.debug("[MPS] Could not patch run_every_op for swap watchdog: %s", e)
        return 0


def _auto_enable_metal_flash_attention():
    """Auto-enable Metal Flash Attention when a suitable backend is available.

    Uses the attention module's probe results to avoid re-checking.
    The probe in attention.py already tested mps-flash-attn (subprocess test)
    and custom Metal FA, setting MPS_FLASH_ATTENTION_IS_AVAILABLE accordingly.
    """
    global _mfa_guard_locked
    if _mfa_guard_locked:
        log.warning("[MPS] Metal Flash Attention auto-enable skipped (guard-locked disabled)")
        return 0

    try:
        from comfy.cli_args import args
        # Don't override if user explicitly chose another attention method
        if any([args.use_split_cross_attention, args.use_quad_cross_attention,
                args.use_pytorch_cross_attention, args.use_sage_attention,
                args.use_flash_attention, args.use_metal_flash_attention]):
            return 0

        import comfy.ldm.modules.attention as attn_mod
        if not attn_mod.MPS_FLASH_ATTENTION_IS_AVAILABLE:
            return 0

        if getattr(attn_mod, "_MFA_NATIVE_BRIDGE_AVAILABLE", False):
            backend = "native MFABridge (metal_sdpa_extension)"
        elif attn_mod._MFA_PACKAGE_AVAILABLE:
            backend = f"mps-flash-attn {attn_mod._mfa.__version__}"
        else:
            backend = "custom Metal FA kernel"
        args.use_metal_flash_attention = True
        _set_mfa_runtime_enabled(True, f'auto-enabled ({backend})')

        # Optional startup prewarm for in-tree Metal FA kernels.
        # Helps remove first-use JIT latency and mimics descriptor registration.
        prewarm_raw = os.environ.get('COMFY_MPS_MFA_PREWARM', '1').strip().lower()
        do_prewarm = prewarm_raw not in {'0', 'false', 'no', 'off'}
        if do_prewarm and not attn_mod._MFA_PACKAGE_AVAILABLE and not getattr(attn_mod, "_MFA_NATIVE_BRIDGE_AVAILABLE", False):
            try:
                import comfy.ldm.modules.metal_attention as metal_attention
                hd_raw = os.environ.get('COMFY_MPS_MFA_PREWARM_HD', '128')
                head_dims = _parse_csv_ints(hd_raw, [128])
                dtypes = [torch.float16]
                if _is_truthy_env('COMFY_MPS_MFA_PREWARM_BF16'):
                    dtypes.append(torch.bfloat16)
                metal_attention.prewarm_metal_flash_attention(head_dims=head_dims, dtypes=dtypes)
            except Exception as e:
                log.warning('[MPS] Metal FA prewarm failed: %s', e)

        log.info("[MPS] Metal Flash Attention auto-enabled (%s) — sub-quad for short, Metal FA for long", backend)
        return 1
    except (ImportError, AttributeError):
        pass
    return 0


def _patch_mfa_generation_guard():
    """Selective MFA crash guard — arm only while MFA kernel code is executing.

    Draw Things pattern (DeviceCapability.swift, ModelPreloader.swift):
    - beginMFAGuard() only arms for untrusted device paths
    - Guard recovery on next launch disables MFA and clears guard

    We arm the persistent guard ONLY around actual MFA kernel dispatch (the
    untrusted path), not at generation start. This way:
    - OOM during model loading → guard NOT armed → MFA stays enabled next launch
    - OOM during sub_quad attention → guard NOT armed → MFA stays enabled
    - Crash during Metal FA kernel → guard IS armed → MFA disabled next launch

    Implementation: arm on first MFA kernel entry per generation, disarm at
    generation end. Single disk write per generation (not per-attention-call).
    """
    if _mfa_guard_locked:
        return 0

    if not _should_arm_mfa_guard():
        log.info("[MPS] MFA guard disabled on trusted platform path (causal-mask capable)")
        return 0

    _mfa_guard_armed = [False]  # mutable container for closures
    patched = []

    def _arm_once(reason):
        if not _mfa_guard_armed[0]:
            _mark_mfa_guard_inflight(True, reason)
            _mfa_guard_armed[0] = True
            log.debug("[MPS] MFA guard armed (%s)", reason)

    def _revert_patches():
        for setter, original in reversed(patched):
            try:
                setter(original)
            except Exception:
                pass
        patched.clear()

    # --- Selective arm: wrap actual in-tree Metal FA dispatch functions ---
    try:
        import comfy.ldm.modules.metal_attention as metal_attn
        dispatch_names = ("_dispatch_v1", "_dispatch_v2", "_dispatch_v3", "_dispatch_v4")
        patched_any_dispatch = False

        for name in dispatch_names:
            if not hasattr(metal_attn, name):
                continue
            original = getattr(metal_attn, name)

            def _make_dispatch_wrapper(fn):
                def wrapped(*args, **kwargs):
                    _arm_once("mfa_dispatch")
                    return fn(*args, **kwargs)
                return wrapped

            wrapped = _make_dispatch_wrapper(original)
            setattr(metal_attn, name, wrapped)
            patched.append((lambda orig, mod=metal_attn, attr=name: setattr(mod, attr, orig), original))
            patched_any_dispatch = True

        if not patched_any_dispatch:
            log.debug("[MPS] No Metal FA dispatch functions found for guard patch")
            return 0
    except (ImportError, AttributeError) as e:
        log.debug("[MPS] Could not patch Metal FA dispatch functions for guard: %s", e)
        return 0

    # --- Selective arm: wrap external mps-flash-attn calls when present ---
    try:
        import comfy.ldm.modules.attention as attn_mod
        native_mod = getattr(attn_mod, "_mfa_native", None)
        if native_mod is not None and hasattr(native_mod, "metal_scaled_dot_product_attention"):
            orig_native = native_mod.metal_scaled_dot_product_attention

            def wrapped_native(*args, **kwargs):
                _arm_once("mfa_native_bridge_dispatch")
                return orig_native(*args, **kwargs)

            native_mod.metal_scaled_dot_product_attention = wrapped_native
            patched.append((lambda orig, mod=native_mod: setattr(mod, "metal_scaled_dot_product_attention", orig), orig_native))

        mfa_mod = getattr(attn_mod, "_mfa", None)
        if mfa_mod is not None:
            if hasattr(mfa_mod, "flash_attention"):
                orig_flash = mfa_mod.flash_attention

                def wrapped_flash(*args, **kwargs):
                    _arm_once("mfa_external_dispatch")
                    return orig_flash(*args, **kwargs)

                mfa_mod.flash_attention = wrapped_flash
                patched.append((lambda orig, mod=mfa_mod: setattr(mod, "flash_attention", orig), orig_flash))

            if hasattr(mfa_mod, "flash_attention_chunked"):
                orig_chunked = mfa_mod.flash_attention_chunked

                def wrapped_chunked(*args, **kwargs):
                    _arm_once("mfa_external_dispatch")
                    return orig_chunked(*args, **kwargs)

                mfa_mod.flash_attention_chunked = wrapped_chunked
                patched.append((lambda orig, mod=mfa_mod: setattr(mod, "flash_attention_chunked", orig), orig_chunked))
    except Exception as e:
        # Non-fatal; external backend is optional and opt-in.
        log.debug("[MPS] Could not patch external mps-flash-attn guard wrappers: %s", e)

    # --- Disarm at generation end: wrap CFGGuider.outer_sample ---
    try:
        import comfy.samplers as samplers
        _orig_outer = samplers.CFGGuider.outer_sample

        def patched_outer(self, *args, **kwargs):
            _mfa_guard_armed[0] = False  # reset per-generation flag
            try:
                return _orig_outer(self, *args, **kwargs)
            finally:
                if _mfa_guard_armed[0]:
                    _mark_mfa_guard_inflight(False, "generation_end")
                    log.debug("[MPS] MFA guard disarmed (generation completed)")
                _mfa_guard_armed[0] = False

        samplers.CFGGuider.outer_sample = patched_outer
        patched.append((lambda orig, mod=samplers.CFGGuider: setattr(mod, "outer_sample", orig), _orig_outer))
    except (ImportError, AttributeError) as e:
        _revert_patches()
        log.debug("[MPS] Could not patch CFGGuider.outer_sample for guard disarm: %s", e)
        return 0

    return 1


def apply_patches():
    """Apply all MPS compatibility patches. Only activates on MPS devices.

    Call this after comfy.options.enable_args_parsing() but before model loading.
    """
    global _patches_applied
    if _patches_applied:
        return

    if not _is_mps_available():
        return

    log.info(
        "[MPS] MFA capability: supported=%s, causal_mask_supported=%s, guard_arming=%s",
        _is_mfa_supported_platform(),
        _is_mfa_causal_mask_supported_platform(),
        _should_arm_mfa_guard(),
    )
    log.info("[MPS] Apple Silicon detected — applying MPS compatibility patches")

    total_patches = 0
    total_patches += _patch_memory_reporting()
    total_patches += _patch_torch_compile()
    total_patches += _patch_model_unload()
    total_patches += _patch_unified_memory_unloading()
    total_patches += _patch_generation_lifecycle()
    total_patches += _patch_swap_watchdog()
    total_patches += _apply_mfa_guardrails()
    total_patches += _auto_enable_metal_flash_attention()
    total_patches += _patch_mfa_generation_guard()

    _patches_applied = True
    log.info("[MPS] Applied {} patches".format(total_patches))
