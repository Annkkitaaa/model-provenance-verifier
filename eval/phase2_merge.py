"""Phase 2: does either verification signal detect a model merge?

Uses mergekit (https://github.com/arcee-ai/mergekit) to produce a linear
(weighted-average) merge of two models already in this project's
known-positive set: gpt2 and its real full fine-tune, lvwerra/gpt2-imdb.
Both verification signals (output-distribution and weight-delta) are then
run against each parent to check whether the merge is correctly flagged as
related to both, which is what a real merge detection use case needs: a
merge is not "related to the base" in the simple fine-tune sense, it is
related to every model that went into it.

mergekit is not in requirements.txt because it is only needed for this one
script, not to reproduce the main Phase 1/2 reports. The PyPI release
(mergekit==0.1.4) has a pydantic v2 compatibility bug that breaks a plain
`pip install mergekit` in this project's environment (ConfiguredModuleArchitecture
is not fully defined); install the current GitHub version instead:

    pip install "git+https://github.com/arcee-ai/mergekit.git"
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from mergekit.config import InputModelDefinition, MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions
except ImportError as exc:  # pragma: no cover - import-time guidance, not logic to test
    raise SystemExit(
        "mergekit is required for this script but is not installed, or the PyPI release's "
        "pydantic bug is breaking it. Install from GitHub: "
        'pip install "git+https://github.com/arcee-ai/mergekit.git"'
    ) from exc

from provenance.model_loading import load_model
from provenance.probes import load_probe_set
from provenance.signals.output_distribution import compute_output_distribution_signal
from provenance.signals.weight_delta import compute_weight_delta_signal

BASE_MODEL_ID = "gpt2"
DERIVATIVE_MODEL_ID = "lvwerra/gpt2-imdb"


def build_merge(out_path: str) -> None:
    config = MergeConfiguration(
        merge_method="linear",
        models=[
            InputModelDefinition(model=BASE_MODEL_ID, parameters={"weight": 0.5}),
            InputModelDefinition(model=DERIVATIVE_MODEL_ID, parameters={"weight": 0.5}),
        ],
        dtype="float32",
    )
    run_merge(config, out_path=out_path, options=MergeOptions(allow_crimes=True, quiet=True))


def main() -> None:
    probe_set = load_probe_set()

    with tempfile.TemporaryDirectory(prefix="provenance_merge_") as merged_dir:
        print(f"Merging {BASE_MODEL_ID} + {DERIVATIVE_MODEL_ID} (linear, 0.5/0.5)...", file=sys.stderr)
        build_merge(merged_dir)

        print("Loading merged model and both parents...", file=sys.stderr)
        lm_merged = load_model(merged_dir)
        lm_base = load_model(BASE_MODEL_ID)
        lm_derivative = load_model(DERIVATIVE_MODEL_ID)

        results = {}
        for parent_name, lm_parent in (("base", lm_base), ("derivative", lm_derivative)):
            wd = compute_weight_delta_signal(lm_parent, lm_merged)
            od = compute_output_distribution_signal(lm_parent, lm_merged, probe_set=probe_set)
            results[parent_name] = {
                "parent_model_id": lm_parent.model_id,
                "weight_delta": wd.to_dict(),
                "output_distribution": od.to_dict(),
            }

    report = {
        "base_model": BASE_MODEL_ID,
        "derivative_model": DERIVATIVE_MODEL_ID,
        "merge_method": "linear",
        "merge_weights": {"base": 0.5, "derivative": 0.5},
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
