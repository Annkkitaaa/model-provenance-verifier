"""Loading for versioned probe sets used by verification signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PROBE_SET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "probes_v1.yaml"
)


@dataclass(frozen=True)
class ProbeSet:
    probe_set_id: str
    prompts: tuple[str, ...]


def load_probe_set(path: str | Path = DEFAULT_PROBE_SET_PATH) -> ProbeSet:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ProbeSet(
        probe_set_id=data["probe_set_id"],
        prompts=tuple(data["prompts"]),
    )
