"""
IO utilities: model/vectorizer/scaler save/load and directory helpers.
"""

import os
import pickle
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def default_model_path(name: str) -> str:
    return os.path.join('models', f'{name}.pkl')


def default_artifact_path(name: str) -> str:
    return os.path.join('artifacts', f'{name}.pkl')
