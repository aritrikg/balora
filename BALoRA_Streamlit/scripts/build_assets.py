"""Export the verified F1 artifacts into the app's compact data formats."""

import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_data import METHODS

ARCHIVE_SHA256 = "ccc8581ea2b5dfac4909056585448a9121f6e1d2a122352f6039d5680c185340"
TABLES = (
    "analysis_summary.json", "validation_histories.json", "failure_auroc_by_volume.csv", "failure_counts_by_volume.csv",
    "replay/variant_summary.csv", "replay/fallback_counts.csv", "replay/fixed_outcome_auroc.csv",
    "replay/paired_dice.csv", "replay/dropout_increment_by_volume.csv", "replay/encoder_timing.csv",
)
COLUMNS = [
    "key", "volume", "organ", "slice", "method", "dice", "iou", "hd95", "assd",
    "ece", "brier", "nll", "ent", "mi",
]


def export_cases(archive, prompts, destination):
    with ZipFile(archive) as z:
        spans = {}
        for method in METHODS:
            prefix = f"/{method}/test/arrays/"
            spans[method] = {
                Path(name).stem: name for name in z.namelist() if prefix in name and name.endswith(".npz")
            }
        keys = set(spans["frozen"])
        assert len(keys) == 18 and all(set(span) == keys for span in spans.values())
        indexed = prompts.set_index(["key", "method"])
        cases = []
        for stem in sorted(keys):
            images, scores = {}, {}
            for method, span in spans.items():
                name = span[stem]
                record = json.loads(z.read(name.replace("/arrays/", "/records/").replace(".npz", ".json")))
                prompt = indexed.loc[record["key"], method]
                assert record["dice"] == prompt.dice or np.isclose(record["dice"], prompt.dice, atol=1e-12, rtol=0)
                ref_key = (record["key"], record["prompt_xy"])
                with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as arr:
                    ref = arr["ground_truth"]
                    ct = arr["windowed_CT"]
                    pred = arr["prediction_FP32_threshold"]
                    assert all(a.shape == (1024, 1024) for a in (ref, ct, pred))
                    if method == "frozen":
                        reference_key = ref_key
                        images["reference"] = ref
                        original_ct = ct
                        images["ct"] = np.rint(ct.astype(np.float32).clip(0, 1) * 255).astype(np.uint8)
                    else:
                        assert ref_key == reference_key
                        np.testing.assert_array_equal(ref, images["reference"])
                        np.testing.assert_array_equal(ct, original_ct)
                    assert set(np.unique(pred)) <= {0, 1} and set(np.unique(ref)) <= {0, 1}
                    actual_dice = 2 * np.count_nonzero(pred & ref) / (int(pred.sum()) + int(ref.sum()))
                    np.testing.assert_allclose(actual_dice, prompt.dice, rtol=0, atol=1e-6)
                    images[method] = pred
                    if method == "balora":
                        images["mi"] = arr["epistemic_MI_FP16"]
                scores[method] = {metric: float(prompt[metric]) for metric in ("dice", "iou")}
            x, y = record["prompt_xy"]
            assert images["reference"][y, x] == 1
            cases.append({
                "key": record["key"], "stem": stem, "volume": record["case_id"],
                "slice": record["slice_index"], "organ": record["organ"], "x": x, "y": y,
                "metrics": scores,
            })
            np.savez_compressed(destination / "examples" / f"{stem}.npz", **images)
    (destination / "case_index.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--archive", type=Path, required=True)
    args.add_argument("--analysis-dir", type=Path, required=True)
    args.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    cli = args.parse_args()
    with cli.archive.open("rb") as archive:
        if hashlib.file_digest(archive, "sha256").hexdigest() != ARCHIVE_SHA256:
            args.error("Provide the completed BALoRA_Full_Run_Results(1).zip archive.")
    cli.output.mkdir(parents=True, exist_ok=True)
    (cli.output / "examples").mkdir(exist_ok=True)
    for name in TABLES:
        target = cli.output / Path(name).name
        shutil.copyfile(cli.analysis_dir / name, target)
    prompts = pd.read_csv(cli.analysis_dir / "per_prompt_fold1_seed42.csv")
    assert len(prompts) == 7398 and prompts.fold.eq(1).all() and prompts.seed.eq(42).all()
    assert prompts.groupby("method").size().eq(2466).all()
    assert prompts.groupby("key").method.nunique().eq(3).all()
    prompts[COLUMNS].sort_values(["volume", "organ", "slice", "method"]).to_csv(
        cli.output / "per_prompt.csv.gz", compression={"method": "gzip", "mtime": 0}, index=False
    )
    export_cases(cli.archive, prompts, cli.output)
    print(f"Exported {len(prompts):,} prediction rows and 18 matched image sets to {cli.output}")


if __name__ == "__main__":
    main()
