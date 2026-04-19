"""
models/loader.py
----------------
Tiny helper so anywhere in the codebase can grab a teammate's model
with one line:

    from models.loader import load_agent
    taylor = load_agent("taylor_cnn", sample=False, strong=True)
"""

from pathlib import Path
from connect4_env import ModelAgent

MODELS_DIR = Path(__file__).parent


def load_model(name: str):
    """
    Load a Keras model by short name (no extension).
    Checks for both .h5 and .keras files.
    """
    from tensorflow import keras  # lazy import — only needed at load time

    for ext in (".h5", ".keras"):
        path = MODELS_DIR / f"{name}{ext}"
        if path.exists():
            return keras.models.load_model(path, compile=False)
    raise FileNotFoundError(
        f"No model named '{name}' in {MODELS_DIR}. "
        f"Available: {[p.name for p in MODELS_DIR.iterdir() if p.suffix in ('.h5', '.keras')]}"
    )


def load_agent(name: str, sample: bool = False, strong: bool = True) -> ModelAgent:
    """
    Load a model and wrap it in a ModelAgent ready for play_game /
    tournament runners.
    """
    model = load_model(name)
    return ModelAgent(model, sample=sample, strong=strong)


def list_available():
    """Return list of model short-names available in this folder."""
    return sorted(
        p.stem for p in MODELS_DIR.iterdir()
        if p.suffix in (".h5", ".keras")
    )
