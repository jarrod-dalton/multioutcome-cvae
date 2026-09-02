"""Portable, versioned checkpoints for fitted CVAE trainers.

The checkpoint is the canonical Python representation of a fitted model.  It
contains primitive metadata, tensor weights, and the fitted X standardizer, but no
optimizer state or training history.  Loading reconstructs the trainer rather
than unpickling a Python model object.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .model import CVAETrainer


FORMAT_VERSION = 1
ARTIFACT_TYPE = "multioutcome-cvae"

PathLike = Union[str, Path]


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Checkpoint metadata field {field!r} must be a positive integer.")
    return value


def _require_finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Checkpoint metadata field {field!r} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"Checkpoint metadata field {field!r} must be {qualifier}.")
    return number


def _normalize_hidden_dims(value: Any, field: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Checkpoint metadata field {field!r} must be a nonempty list.")
    return [_require_positive_int(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _normalize_schema(value: Any, y_dim: int) -> List[Dict[str, Any]]:
    """Validate a serialized categorical schema without trusting model internals."""
    if not isinstance(value, list) or len(value) != y_dim:
        raise ValueError(
            "Categorical checkpoint outcome_schema must be a list with one entry "
            "per semantic outcome."
        )

    schema: List[Dict[str, Any]] = []
    seen_names = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Categorical outcome_schema[{index}] must be an object.")
        name = entry.get("name")
        levels = entry.get("levels")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Categorical outcome_schema[{index}].name must be a nonempty string."
            )
        if name in seen_names:
            raise ValueError(f"Categorical outcome_schema contains duplicate name {name!r}.")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(
                f"Categorical outcome_schema[{index}].levels must contain at least two levels."
            )
        if any(not isinstance(level, str) or not level.strip() for level in levels):
            raise ValueError(
                f"Categorical outcome_schema[{index}].levels must be nonempty strings."
            )
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"Categorical outcome_schema[{index}].levels must be unique."
            )
        seen_names.add(name)
        schema.append({"name": name, "levels": list(levels)})
    return schema


def _derive_categorical_layout(
    schema: List[Dict[str, Any]],
) -> Tuple[int, List[List[int]]]:
    slices: List[List[int]] = []
    start = 0
    for entry in schema:
        stop = start + len(entry["levels"])
        slices.append([start, stop])
        start = stop
    return start, slices


def _normalize_slices(value: Any) -> Optional[List[List[int]]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Categorical outcome_slices must be a sequence of pairs.")
    normalized: List[List[int]] = []
    for index, item in enumerate(value):
        if isinstance(item, slice):
            if item.step not in (None, 1):
                raise ValueError(f"Categorical outcome_slices[{index}] has an invalid step.")
            pair = [item.start, item.stop]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pair = list(item)
        else:
            raise ValueError(f"Categorical outcome_slices[{index}] must be a start/stop pair.")
        if any(isinstance(bound, bool) or not isinstance(bound, int) for bound in pair):
            raise ValueError(f"Categorical outcome_slices[{index}] bounds must be integers.")
        normalized.append(pair)
    return normalized


def _categorical_metadata(trainer: CVAETrainer) -> Dict[str, Any]:
    schema = _normalize_schema(deepcopy(getattr(trainer, "outcome_schema", None)), trainer.y_dim)
    encoded_y_dim, outcome_slices = _derive_categorical_layout(schema)

    # Guard against saving an internally inconsistent fitted object.
    for owner_name, owner in (("trainer", trainer), ("model", trainer.model)):
        stored_schema = _normalize_schema(
            deepcopy(getattr(owner, "outcome_schema", None)), trainer.y_dim
        )
        stored_dim = getattr(owner, "encoded_y_dim", None)
        stored_slices = _normalize_slices(getattr(owner, "outcome_slices", None))
        if stored_schema != schema:
            raise RuntimeError(
                f"Cannot save categorical checkpoint: {owner_name}.outcome_schema "
                "does not match the fitted trainer schema."
            )
        if stored_dim != encoded_y_dim:
            raise RuntimeError(
                f"Cannot save categorical checkpoint: {owner_name}.encoded_y_dim "
                "does not match outcome_schema."
            )
        if stored_slices != outcome_slices:
            raise RuntimeError(
                f"Cannot save categorical checkpoint: {owner_name}.outcome_slices "
                "does not match outcome_schema."
            )

    return {
        "type": "categorical",
        "outcome_schema": schema,
        "encoded_y_dim": encoded_y_dim,
        "outcome_slices": outcome_slices,
    }


def _build_metadata(trainer: CVAETrainer) -> Dict[str, Any]:
    if trainer.outcome_type == "categorical":
        outcome = _categorical_metadata(trainer)
    else:
        outcome = {
            "type": trainer.outcome_type,
            "outcome_schema": None,
            "encoded_y_dim": int(trainer.y_dim),
            "outcome_slices": None,
        }

    return {
        "artifact_type": ARTIFACT_TYPE,
        "architecture": {
            "x_dim": int(trainer.x_dim),
            "y_dim": int(trainer.y_dim),
            "latent_dim": int(trainer.latent_dim),
            # Record the architecture that was actually instantiated. Legacy
            # callers may pass empty lists, which the model normalizes even
            # though the trainer retains the original constructor values.
            "enc_hidden_dims": [
                int(layer.out_features) for layer in trainer.model.enc_layers
            ],
            "dec_hidden_dims": [
                int(layer.out_features) for layer in trainer.model.dec_layers
            ],
        },
        "trainer": {
            "num_epochs": int(trainer.num_epochs),
            "batch_size": int(trainer.batch_size),
            "lr": float(trainer.lr),
            "beta_kl": float(trainer.beta_kl),
        },
        "outcome": outcome,
    }


def _standardizer_tensor(value: Any, field: str, x_dim: int) -> torch.Tensor:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot save checkpoint: {field} is invalid.") from exc
    if array.shape != (x_dim,) or not np.isfinite(array).all():
        raise RuntimeError(
            f"Cannot save checkpoint: {field} must be a finite vector of length {x_dim}."
        )
    if field == "x_std" and np.any(array <= 0.0):
        raise RuntimeError("Cannot save checkpoint: x_std must be strictly positive.")
    return torch.from_numpy(array.copy())


def save_cvae(trainer: CVAETrainer, output_path: PathLike) -> Path:
    """Save a fitted trainer as a portable version-1 inference checkpoint."""
    if not isinstance(trainer, CVAETrainer):
        raise TypeError("trainer must be a CVAETrainer instance.")
    if not trainer.trained:
        raise RuntimeError("Model must be fitted before it can be saved.")

    metadata = _build_metadata(trainer)
    # Apply the same strict metadata validation used at load time before any
    # bytes are written.
    _parse_configuration(metadata)
    payload = {
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.state_dict().items()
        },
        "x_mean": _standardizer_tensor(trainer.x_mean, "x_mean", trainer.x_dim),
        "x_std": _standardizer_tensor(trainer.x_std, "x_std", trainer.x_dim),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _safe_torch_load(path: Path, map_location: torch.device) -> Any:
    """Prefer restricted loading while retaining compatibility with torch 1.x."""
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    # ``weights_only`` was added after some supported historical installs.
    # Select this compatibility path before parsing; never retry unrestricted
    # loading after a restricted load has begun.
    return torch.load(path, map_location=map_location)


def _parse_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    metadata = deepcopy(dict(value))
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Checkpoint metadata has an invalid artifact_type.")
    return metadata


def _parse_configuration(metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    architecture = metadata.get("architecture")
    training = metadata.get("trainer")
    outcome = metadata.get("outcome")
    if not isinstance(architecture, dict) or not isinstance(training, dict):
        raise ValueError("Checkpoint metadata is missing architecture or trainer configuration.")
    if not isinstance(outcome, dict):
        raise ValueError("Checkpoint metadata is missing outcome configuration.")

    config = {
        "x_dim": _require_positive_int(architecture.get("x_dim"), "architecture.x_dim"),
        "y_dim": _require_positive_int(architecture.get("y_dim"), "architecture.y_dim"),
        "latent_dim": _require_positive_int(
            architecture.get("latent_dim"), "architecture.latent_dim"
        ),
        "enc_hidden_dims": _normalize_hidden_dims(
            architecture.get("enc_hidden_dims"), "architecture.enc_hidden_dims"
        ),
        "dec_hidden_dims": _normalize_hidden_dims(
            architecture.get("dec_hidden_dims"), "architecture.dec_hidden_dims"
        ),
        "num_epochs": _require_positive_int(training.get("num_epochs"), "trainer.num_epochs"),
        "batch_size": _require_positive_int(training.get("batch_size"), "trainer.batch_size"),
        "lr": _require_finite_number(training.get("lr"), "trainer.lr", positive=True),
        "beta_kl": _require_finite_number(training.get("beta_kl"), "trainer.beta_kl"),
    }
    if config["beta_kl"] < 0.0:
        raise ValueError("Checkpoint metadata field 'trainer.beta_kl' must be non-negative.")

    outcome_type = outcome.get("type")
    if not isinstance(outcome_type, str) or not outcome_type:
        raise ValueError("Checkpoint outcome type must be a nonempty string.")
    config["outcome_type"] = outcome_type

    if outcome_type == "categorical":
        schema = _normalize_schema(outcome.get("outcome_schema"), config["y_dim"])
        encoded_y_dim, outcome_slices = _derive_categorical_layout(schema)
        if outcome.get("encoded_y_dim") != encoded_y_dim:
            raise ValueError(
                "Categorical checkpoint encoded_y_dim does not match outcome_schema."
            )
        if _normalize_slices(outcome.get("outcome_slices")) != outcome_slices:
            raise ValueError(
                "Categorical checkpoint outcome_slices do not match outcome_schema."
            )
        config["outcome_schema"] = schema
    else:
        if outcome.get("outcome_schema") is not None:
            raise ValueError("Non-categorical checkpoint must not contain outcome_schema.")
        if outcome.get("encoded_y_dim") != config["y_dim"]:
            raise ValueError("Checkpoint encoded_y_dim does not match y_dim.")
        if outcome.get("outcome_slices") is not None:
            raise ValueError("Non-categorical checkpoint must not contain outcome_slices.")

    return config, outcome


def _load_standardizer(value: Any, field: str, x_dim: int) -> np.ndarray:
    if not torch.is_tensor(value):
        raise ValueError(f"Checkpoint field {field!r} must be a tensor.")
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tuple(tensor.shape) != (x_dim,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(
            f"Checkpoint field {field!r} must be a finite vector of length {x_dim}."
        )
    if field == "x_std" and not bool(torch.all(tensor > 0.0)):
        raise ValueError("Checkpoint field 'x_std' must be strictly positive.")
    return tensor.numpy().copy()


def load_cvae(path: PathLike, device: Optional[Union[str, torch.device]] = None) -> CVAETrainer:
    """Load and validate a version-1 CVAE inference checkpoint."""
    requested_device = torch.device(device) if device is not None else torch.device("cpu")
    payload = _safe_torch_load(Path(path), requested_device)
    if not isinstance(payload, Mapping):
        raise ValueError("CVAE checkpoint must contain a mapping.")

    version = payload.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("CVAE checkpoint format_version must be an integer.")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported CVAE checkpoint format_version {version}; expected {FORMAT_VERSION}."
        )

    metadata = _parse_metadata(payload.get("metadata"))
    config, outcome = _parse_configuration(metadata)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint state_dict must be a nonempty mapping.")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state_dict.items()):
        raise ValueError("Checkpoint state_dict must map string names to tensors.")

    constructor_args = dict(config)
    if config["outcome_type"] != "categorical":
        constructor_args.pop("outcome_schema", None)
    trainer = CVAETrainer(device=str(requested_device), **constructor_args)

    if config["outcome_type"] == "categorical":
        expected_schema = config["outcome_schema"]
        expected_dim, expected_slices = _derive_categorical_layout(expected_schema)
        actual_schema = _normalize_schema(
            deepcopy(getattr(trainer, "outcome_schema", None)), config["y_dim"]
        )
        actual_slices = _normalize_slices(getattr(trainer, "outcome_slices", None))
        if (
            actual_schema != expected_schema
            or getattr(trainer, "encoded_y_dim", None) != expected_dim
            or actual_slices != expected_slices
        ):
            raise ValueError(
                "Reconstructed categorical model layout does not match checkpoint metadata."
            )
        # The serialized values were already checked, but retaining this explicit
        # comparison makes the checkpoint contract visible at the restore boundary.
        if outcome["encoded_y_dim"] != expected_dim:
            raise ValueError("Categorical checkpoint layout validation failed.")

    trainer.x_mean = _load_standardizer(payload.get("x_mean"), "x_mean", trainer.x_dim)
    trainer.x_std = _load_standardizer(payload.get("x_std"), "x_std", trainer.x_dim)
    try:
        trainer.model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint state_dict is incompatible with its metadata.") from exc
    trainer.model.eval()
    trainer.trained = True
    return trainer


__all__ = ["save_cvae", "load_cvae"]
