"""
models/loader.py
----------------
Tiny helper so anywhere in the codebase can grab a teammate's model
with one line, regardless of whether it was trained in Keras or PyTorch.

    from models.loader import load_agent
    taylor = load_agent("taylor_cnn")      # auto-detects .h5 / .keras / .pt / .pth
"""

from pathlib import Path
from connect4_env import ModelAgent

MODELS_DIR = Path(__file__).parent

KERAS_EXTS = (".h5", ".keras")
TORCH_EXTS = (".pt", ".pth")


def _find_model_file(name: str) -> Path:
    """Find a model file by short name, checking all supported extensions."""
    for ext in KERAS_EXTS + TORCH_EXTS:
        path = MODELS_DIR / f"{name}{ext}"
        if path.exists():
            return path
    available = [p.name for p in MODELS_DIR.iterdir()
                 if p.suffix in KERAS_EXTS + TORCH_EXTS]
    raise FileNotFoundError(
        f"No model named '{name}' in {MODELS_DIR}. Available: {available}"
    )


def load_model(name: str):
    """
    Load a model by short name (no extension). Returns the raw model
    (Keras or PyTorch) — use `load_agent()` if you want it ready-to-play.
    """
    path = _find_model_file(name)

    if path.suffix in KERAS_EXTS:
        from tensorflow import keras  # lazy import
        return keras.models.load_model(path, compile=False)

    if path.suffix in TORCH_EXTS:
        import torch  # lazy import
        # torch.load returns either a full nn.Module (if saved with
        # torch.save(model, path)) or a state_dict. We need the former.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, torch.nn.Module):
            raise TypeError(
                f"{path.name} appears to be a state_dict, not a full model. "
                "Please re-save using `torch.save(model, path)` rather than "
                "`torch.save(model.state_dict(), path)` — we need the full "
                "model object to run inference without the architecture code."
            )
        obj.eval()
        return obj

    raise ValueError(f"Unsupported model extension: {path.suffix}")


def load_agent(name: str, sample: bool = False, strong: bool = True,
               channels_first: bool = True) -> ModelAgent:
    """
    Load a model and wrap it in a ModelAgent ready for play_game /
    tournament runners.

    channels_first only matters for PyTorch models. Default True matches
    the typical PyTorch convention (N, C, H, W). If your PyTorch model
    was trained with channels-last inputs, pass channels_first=False.
    """
    model = load_model(name)
    return ModelAgent(model, sample=sample, strong=strong,
                      channels_first=channels_first)


def list_available():
    """Return list of model short-names available in this folder."""
    return sorted(
        p.stem for p in MODELS_DIR.iterdir()
        if p.suffix in KERAS_EXTS + TORCH_EXTS
    )
