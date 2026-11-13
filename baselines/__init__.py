"""NeuroSAGE baselines: configurable training / evaluation pipeline for brain-tumour MRI classification."""
import sys
import warnings

# Windows consoles default to cp1252, which cannot print '±' / 'Δ'.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# With AMP the GradScaler may skip the very first optimiser step; the scheduler warning about it is harmless.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")
