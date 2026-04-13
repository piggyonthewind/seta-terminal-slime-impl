"""
sitecustomize.py — executed automatically by Python on startup (from PYTHONPATH).
Caps torch_memory_saver.memory_margin_bytes so single-GPU 8 GB training can fit
the Megatron DDP param + grad buffers within available VRAM.

Without this cap, SLIME's actor.py hard-sets the margin to 1 GiB (the default),
which blocks the bf16 grad-buffer allocation (6.42 GB) when only ~6.66 GB is free
after the SGLang KV-cache pool reduction.

The cap is controlled by TMS_MAX_MARGIN_BYTES (default: 128 MiB = 134217728).
"""
import os
import sys

_MAX_MARGIN = int(os.environ.get("TMS_MAX_MARGIN_BYTES", 16 * 1024 * 1024))  # 16 MiB


def _install_tms_margin_cap():
    try:
        import torch_memory_saver as _tms_pkg
        _singleton = _tms_pkg.torch_memory_saver
        _cls = type(_singleton)

        _orig_prop = _cls.memory_margin_bytes
        _orig_fset = _orig_prop.fset

        def _capped_fset(self, value):
            capped = min(int(value), _MAX_MARGIN)
            if capped != int(value):
                import logging
                logging.getLogger("sitecustomize").info(
                    f"[sitecustomize] TorchMemorySaver.memory_margin_bytes capped: "
                    f"{value} → {capped} (TMS_MAX_MARGIN_BYTES={_MAX_MARGIN})"
                )
            _orig_fset(self, capped)

        _cls.memory_margin_bytes = property(
            _orig_prop.fget,
            _capped_fset,
            _orig_prop.fdel,
        )
    except Exception as e:
        # Non-fatal: if TMS isn't available yet or the hook fails, carry on.
        pass


_install_tms_margin_cap()
