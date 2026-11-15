from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from study_data import METHODS, PAIR, PROMPT, aggregate, case_view, read_case, read_json, read_table

APP = Path(__file__).resolve().parents[1] / "app.py"


class DataTests(TestCase):
    def test_published_means_and_matched_cohort(self):
        prompts = read_table("per_prompt.csv.gz")
        self.assertEqual(len(prompts), 7398)
        self.assertEqual(prompts.key.nunique(), 2466)
        self.assertEqual(prompts.volume.nunique(), 6)
        self.assertEqual(prompts.organ.nunique(), 13)
        self.assertEqual(prompts[["volume", "organ"]].drop_duplicates().shape[0], 78)
        self.assertTrue(prompts.groupby("key").method.nunique().eq(3).all())
        summary = read_json("analysis_summary.json")
        for weighting, expected in ((PAIR, "dice_pair_weighted"), (PROMPT, "dice_prompt_weighted")):
            for row in aggregate(prompts, ["method"], "dice", weighting).itertuples():
                self.assertAlmostEqual(row.dice, summary[expected][row.method], places=12)

    def test_case_masks_and_shared_prompt(self):
        cases = read_json("case_index.json")
        self.assertEqual(len(cases), 18)
        for case in cases:
            arrays = read_case(case["stem"])
            ref = arrays["reference"].astype(bool)
            self.assertEqual(ref.shape, (1024, 1024))
            self.assertTrue(ref[case["y"], case["x"]])
            self.assertTrue(np.isfinite(arrays["mi"]).all())
            for method in METHODS:
                mask = arrays[method].astype(bool)
                overlap = (mask & ref).sum()
                self.assertAlmostEqual(2 * overlap / (mask.sum() + ref.sum()),
                                       case["metrics"][method]["dice"], places=6)
        image = case_view(arrays, case["x"], case["y"], "mi", "Contours", True, True, True)
        self.assertEqual(image.mode, "RGB")
        self.assertLessEqual(max(image.size), 1024)


class AppTests(TestCase):
    def start(self):
        return self.clean(AppTest.from_file(str(APP), default_timeout=25).run())

    def clean(self, app):
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_benchmark_filter_and_weighting(self):
        app = self.start()
        self.assertEqual([m.value for m in app.metric], ["0.5123", "0.8049", "0.7982"])
        self.assertTrue(app.get("download_button"))
        app.multiselect[0].select("liver")
        app.multiselect[1].select("img0034")
        app.selectbox(key="metric").select("IoU")
        app.selectbox(key="weighting").select(PROMPT)
        app.selectbox(key="breakdown").select("Volume")
        self.clean(app.run())
        self.assertEqual(list(app.dataframe[0].value.columns), ["Method", "IoU"])

    def test_ct_methods_and_mi(self):
        app = self.start()
        self.clean(app.radio(key="page").set_value("CT cases").run())
        self.assertEqual(len(app.get("image")), 4)
        app.selectbox(key="case_organ").select("pancreas")
        app.multiselect(key="case_methods").set_value(["balora"])
        app.radio(key="overlay").set_value("Mask errors")
        app.toggle(key="show_mi").set_value(True)
        self.clean(app.run())
        self.assertEqual(len(app.get("image")), 3)

    def test_replay_views(self):
        app = self.start()
        self.clean(app.radio(key="page").set_value("Inference replay").run())
        self.assertEqual(len(app.dataframe[0].value), 8)
        self.clean(app.radio(key="replay_section").set_value("Failure ranking").run())
        self.assertEqual([m.value for m in app.metric], ["0.8222", "0.8238", "+0.0016"])
        self.clean(app.selectbox(key="replay_comparison").select(
            "Gaussian MI added to organ, area and entropy").run())
        self.assertEqual([m.value for m in app.metric], ["0.8259", "0.8257", "-0.0001"])
        self.clean(app.radio(key="replay_section").set_value("Mask fallback").run())
        self.assertEqual(len(app.dataframe[0].value), 7)
        self.clean(app.selectbox(key="fallback_volume").select("img0009").run())
        self.assertEqual(len(app.dataframe[0].value), 7)
        self.clean(app.radio(key="replay_section").set_value("Encoder timing").run())
        self.assertEqual(len(app.dataframe[0].value), 12)

    def test_protocol_and_metrics_only_mode(self):
        app = self.start()
        self.clean(app.radio(key="page").set_value("Study protocol").run())
        self.assertEqual([m.value for m in app.metric], ["19", "5", "6"])
        with patch("pathlib.Path.is_file", autospec=True, return_value=False):
            self.clean(app.radio(key="page").set_value("CT cases").run())
        self.assertIn("aggregate results", app.info[0].value)
