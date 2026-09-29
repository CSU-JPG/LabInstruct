"""Adapter factory (lazy loading).

``build_adapter(model_id, model_cfg, root)`` returns an adapter instance.
Adapter modules import heavy dependencies (torch, diffusers, model repos)
lazily, so the lightweight harness env only needs PyYAML/requests/Pillow; the
heavy modules are imported when a model actually runs (in its own env).

Model id -> adapter module mapping (one module may serve several ids):
    ltx2.3                  -> ltx
    wan2.2                  -> wan
    cosmos3-nano / -super   -> cosmos
    lingbot-video           -> lingbot
    minimax-h3              -> minimax
"""
from __future__ import annotations

import importlib
import pathlib
from typing import Any

from bench.adapters.base import BaseAdapter

_MODULE_BY_MODEL_ID = {
    "ltx2.3": "ltx",
    "wan2.2": "wan",
    "cosmos3-nano": "cosmos",
    "cosmos3-super": "cosmos",
    "lingbot-video": "lingbot",
    "minimax-h3": "minimax",
}

_CLASS_BY_MODEL_ID = {
    "ltx2.3": "LTXAdapter",
    "wan2.2": "WanAdapter",
    "cosmos3-nano": "CosmosAdapter",
    "cosmos3-super": "CosmosAdapter",
    "lingbot-video": "LingBotAdapter",
    "minimax-h3": "MiniMaxAdapter",
}


def build_adapter(
    model_id: str,
    model_cfg: dict[str, Any],
    root: pathlib.Path,
) -> BaseAdapter:
    if model_id not in _MODULE_BY_MODEL_ID:
        raise KeyError(
            f"no adapter registered for model id {model_id!r}; "
            f"known: {sorted(_MODULE_BY_MODEL_ID)}"
        )
    module = importlib.import_module(
        f"bench.adapters.{_MODULE_BY_MODEL_ID[model_id]}"
    )
    cls = getattr(module, _CLASS_BY_MODEL_ID[model_id])
    adapter = cls(model_cfg, root)
    # One class may serve several registry keys (e.g. Cosmos3-Nano/Super);
    # stamp the actual registry key so the adapter knows which variant it is.
    adapter.model_id = model_id
    return adapter
