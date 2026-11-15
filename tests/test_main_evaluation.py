"""Research runner integration/metric checks, not paper evaluation results."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments import main_evaluation as study


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = next(study.encounters(314159,1))

    def test_classification_truth_covers_full_encounter(self):
        distances,_,_=study.stationary_geometry(np.array(self.case["state"]),
                                               np.array(self.case["uav_positions"]),self.case["duration_s"])
        truth=distances<study.PROJECTILE_CLEARANCE
        correct=study.classification_metrics(self.case,truth)
        self.assertEqual(correct["hit_recall"],1)
        self.assertEqual(correct["false_positive_rate"],0)
        inverted=study.classification_metrics(self.case,~truth)
        self.assertEqual(inverted["hit_recall"],0)
        self.assertEqual(inverted["false_positive_rate"],1)
        self.assertIsNone(study.classification_metrics(self.case,None)["hit_recall"])

    def test_baseline_receives_exact_saved_encounter(self):
        def baseline(**kwargs):
            np.testing.assert_array_equal(kwargs["projectile_state"],self.case["state"])
            np.testing.assert_array_equal(kwargs["uav_positions"],self.case["uav_positions"])
            self.assertEqual(kwargs["duration"],self.case["duration_s"])
            self.assertEqual(kwargs["seed"],123)
            return SimpleNamespace(min_projectile_distance=.2,min_uav_distance=.3,total_path_length=.45,
                                   number_moved=1,path_lengths=(.45,0,0,0),predictions=("hit","miss","miss","miss"),
                                   first_response_time=.2,decision_latency=.2,response_delay=0,
                                   detection_compute_s=.001,response_compute_s=.0001,acoustic_generation_s=.01,
                                   first_motion_time=.204)
        row=study.run_dopplerdodge(self.case,10,123,SimpleNamespace(run_baseline=baseline))
        self.assertTrue(row["collision_free"])
        self.assertEqual(row["number_moved"],1)
        self.assertEqual(row["avoidance_response_time_s"],0)
        self.assertEqual(row["baseline_seed"],123)
        self.assertIsNone(row["position_rmse_m"])
        self.assertIsNone(row["initialization_success"])

    def test_summary_rates_errors_and_missing_legacy_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            (output/"manifest.json").write_text(json.dumps({"config":{"methods":list(study.METHODS)}}))
            (output/"encounters.jsonl").write_text(json.dumps(self.case)+"\n")
            good=study.static_result(self.case)
            good.update(safety_satisfied=True,projectile_collision=False,uav_collision=False)
            rows=[dict(encounter_id=self.case["encounter_id"],method="dopplerdodge",**good),
                  dict(encounter_id=self.case["encounter_id"],method="estimate",status="error",safety_satisfied=False)]
            (output/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                study.summarize(output)
            with (output/"summary.csv").open(newline="") as file:
                summaries={r["method"]:r for r in csv.DictReader(file)}
            self.assertEqual(float(summaries["dopplerdodge"]["collision_free_success_rate"]),1)
            self.assertEqual(float(summaries["estimate"]["collision_free_success_rate"]),0)
            self.assertEqual(summaries["estimate"]["projectile_collision_rate"],"")
            self.assertEqual(summaries["estimate"]["number_moved_observations"],"0")
            self.assertIn("/ 4 method trials",(output/"report.md").read_text(encoding="utf-8"))

    def test_cli_calibrates_once_and_resumes_without_repeating_methods(self):
        from baselines import dopplerdodge
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"run"
            arguments=["main_evaluation","--output",str(output),"--repetitions","1"]
            def online(case,*args):
                return study.static_result(case)
            with patch.object(study,"encounters",return_value=iter([self.case])), \
                 patch.object(study,"run_online",side_effect=online) as run_online, \
                 patch.object(study,"run_dopplerdodge",side_effect=online) as run_dodge, \
                 patch.object(dopplerdodge,"calibrate_threshold",return_value=12.5) as calibrate, \
                 patch.object(sys,"argv",arguments),contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(calibrate.call_count,1)
            self.assertEqual(run_online.call_count,2)
            self.assertEqual(run_dodge.call_count,1)
            manifest=json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["dopplerdodge_threshold"],12.5)
            self.assertIn("baselines\\dopplerdodge.py" if sys.platform=="win32" else "baselines/dopplerdodge.py",manifest["source_hashes"])
            with patch.object(study,"run_online",side_effect=AssertionError("reran")), \
                 patch.object(study,"run_dopplerdodge",side_effect=AssertionError("reran")), \
                 patch.object(dopplerdodge,"calibrate_threshold",side_effect=AssertionError("recalibrated")), \
                 patch.object(sys,"argv",arguments+["--resume"]),contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(len(study.load_jsonl(output/"results.jsonl")),4)

    def test_reuse_legacy_results_runs_only_new_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary)/"parent"
            parent.mkdir()
            cases_text=json.dumps(self.case)+"\n"
            (parent/"encounters.jsonl").write_text(cases_text,encoding="utf-8")
            legacy_config=dict(version=1,seed=314159,repetitions=1,sizes=list(study.SIZES),speeds=list(study.SPEEDS),
                               classes=list(study.CLASSES),methods=["static","oracle","estimate"],rms_radius_m=1.0,
                               target_time_s=study.WARNING,uplink_s=0.0,downlink_s=0.0,added_noise=False)
            hashes={k:v for k,v in study.source_hashes().items() if not k.startswith("baselines")}
            manifest=dict(config=legacy_config,source_hashes=hashes,
                          environment=json.loads(json.dumps(study.environment())),
                          encounters_sha256=study.hashlib.sha256((parent/"encounters.jsonl").read_bytes()).hexdigest())
            (parent/"manifest.json").write_text(json.dumps(manifest))
            result=study.static_result(self.case)
            result.pop("metric_schema_version")
            (parent/"results.jsonl").write_text("".join(json.dumps(dict(encounter_id=self.case["encounter_id"],method=m,**result))+"\n"
                                                       for m in legacy_config["methods"]))
            original=(parent/"results.jsonl").read_bytes()
            output=Path(temporary)/"new"
            args=["main_evaluation","--output",str(output),"--reuse-results",str(parent),
                  "--repetitions","1","--dodge-threshold","10"]
            with patch.object(study,"run_online",side_effect=AssertionError("reran existing method")), \
                 patch.object(study,"run_dopplerdodge",return_value=study.static_result(self.case)) as dodge, \
                 patch.object(sys,"argv",args),contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(dodge.call_count,1)
            rows=study.load_jsonl(output/"results.jsonl")
            self.assertEqual(len(rows),4)
            self.assertEqual(sum(r.get("origin")=="imported" for r in rows),3)
            self.assertEqual((parent/"results.jsonl").read_bytes(),original)


if __name__=="__main__":
    unittest.main()
