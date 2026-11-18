"""Checks for the exploratory scaling protocol, not paper results."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.scaling import runner as study


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=list(study.generate_encounters(study.SIZES,study.RADII,3,57721,3,1.2))

    def test_full_default_design_and_actual_classes(self):
        self.assertEqual(len(self.cases),108)
        self.assertEqual(len({c["encounter_id"] for c in self.cases}),108)
        counts=Counter((c["swarm_size"],c["formation_radius_m"],c["category"]) for c in self.cases)
        self.assertEqual(set(counts.values()),{3})
        for case in self.cases:
            positions=np.asarray(case["uav_positions"])
            state=np.asarray(case["state"])
            self.assertAlmostEqual(np.sqrt(np.mean(np.sum(positions**2,axis=1))),case["formation_radius_m"])
            self.assertGreater(case["initial_min_spacing_m"],study.main.UAV_CLEARANCE)
            distances,_,_=study.main.stationary_geometry(state,positions,case["duration_s"])
            self.assertEqual(study.main.classify(distances),case["category"])
            bound=max(np.linalg.norm(positions,axis=1))+study.main.PROJECTILE_CLEARANCE
            self.assertGreater(np.linalg.norm(state[:3]),bound)
            end=state[:3]+case["duration_s"]*state[3:]
            self.assertGreater(np.linalg.norm(end),bound)
            self.assertGreater(end@state[3:],0)

    def test_radius_pairing_and_determinism(self):
        regenerated=list(study.generate_encounters(study.SIZES,study.RADII,3,57721,3,1.2))
        self.assertEqual(self.cases,regenerated)
        paired={}
        for case in self.cases:
            normalized=np.asarray(case["uav_positions"])/case["formation_radius_m"]
            if case["geometry_id"] in paired:
                np.testing.assert_allclose(normalized,paired[case["geometry_id"]])
            paired[case["geometry_id"]]=normalized
        self.assertEqual(len(paired),36)

    def test_invalid_initial_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"initial spacing"):
            list(study.generate_encounters([100],[.1],1,7,3,1.2))

    def test_timeout_is_unknown_safety_not_a_collision(self):
        result=study.run_bounded(self.cases[0],"estimate",10,7,.001)
        self.assertEqual(result["status"],"timeout")
        self.assertIsNone(result["safety_satisfied"])
        self.assertNotIn("projectile_collision",result)

    def test_cli_checkpoints_resumes_and_reports_unknowns(self):
        with tempfile.TemporaryDirectory(prefix="scaling_protocol_check_") as temporary:
            output=Path(temporary)/"study"
            args=["experiments.scaling","--output",str(output),"--dodge-threshold","10",
                  "--swarm-sizes","10","25","--radii","1","2","--repetitions","1"]
            def mock_run(case,method,*args):
                if method=="estimate":
                    return dict(status="timeout",safety_satisfied=None,error="development stub")
                return study.main.static_result(case)
            with patch.object(sys,"argv",args),patch.object(study,"run_bounded",side_effect=mock_run), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            rows=study.main.load_jsonl(output/"results.jsonl")
            self.assertEqual(len(rows),48)
            self.assertEqual({r["method"] for r in rows},set(study.METHODS))
            self.assertTrue((output/"source.zip").exists())
            with (output/"summary.csv").open(newline="") as file:
                summaries=list(csv.DictReader(file))
            estimate=next(s for s in summaries if s["method"]=="estimate")
            self.assertEqual(estimate["observed_safety_rate"],"")
            self.assertEqual(estimate["projectile_collision_rate"],"")
            self.assertEqual(estimate["timeouts"],"1")
            with patch.object(sys,"argv",args+["--resume"]), \
                 patch.object(study,"run_bounded",side_effect=AssertionError("reran saved trial")), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            self.assertEqual(len(study.main.load_jsonl(output/"results.jsonl")),48)

    def test_reuse_previous_scaling_runs_only_new_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/"source"
            args=["experiments.scaling","--output",str(source),"--dodge-threshold","10",
                  "--swarm-sizes","10","--radii","1","--repetitions","1"]
            with patch.object(sys,"argv",args),patch.object(study,"run_bounded",side_effect=lambda c,*args:study.main.static_result(c)), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            manifest=json.loads((source/"manifest.json").read_text())
            manifest["config"]["version"]=1
            manifest["config"]["methods"]=list(study.METHODS[:-1])
            manifest["planned_trials"]=9
            (source/"manifest.json").write_text(json.dumps(manifest))
            old_rows=[r for r in study.main.load_jsonl(source/"results.jsonl") if r["method"]!="dopplerdodge_coordinated"]
            (source/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in old_rows))
            original=(source/"results.jsonl").read_bytes()
            output=root/"new"
            args[2]=str(output)
            args += ["--reuse-results",str(source)]
            with patch.object(sys,"argv",args),patch.object(study,"run_bounded",side_effect=lambda c,*args:study.main.static_result(c)) as run, \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            self.assertEqual(run.call_count,3)
            self.assertEqual({call.args[1] for call in run.call_args_list},{"dopplerdodge_coordinated"})
            rows=study.main.load_jsonl(output/"results.jsonl")
            self.assertEqual(len(rows),12)
            self.assertEqual(sum(r.get("origin")=="imported" for r in rows),9)
            self.assertEqual((source/"results.jsonl").read_bytes(),original)


if __name__=="__main__":
    unittest.main()
