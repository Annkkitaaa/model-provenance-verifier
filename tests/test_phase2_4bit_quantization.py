import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from phase2_4bit_quantization import _score_from_file


def test_score_from_file_reads_per_pair_format(tmp_path):
    path = tmp_path / "phase1.json"
    path.write_text(
        json.dumps({"per_pair": [{"model_a": "gpt2", "model_b": "lvwerra/gpt2-imdb", "score": 0.869}]}),
        encoding="utf-8",
    )
    assert _score_from_file(path, "unused") == 0.869


def test_score_from_file_reads_flat_key_format(tmp_path):
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps({"phase2_score_fp32_vs_int8_dynamic": 0.751}), encoding="utf-8")
    assert _score_from_file(path, "phase2_score_fp32_vs_int8_dynamic") == 0.751


def test_score_from_file_returns_none_when_missing(tmp_path):
    assert _score_from_file(tmp_path / "does_not_exist.json", "key") is None
