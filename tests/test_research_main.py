"""Protocol, calibration, bookkeeping and process checks; not paper results."""

import contextlib
import csv
import io
import json
import tempfile
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments import research_main as study


def configuration(**updates):
    config=dict(sizes=list(study.SIZES),speeds=list(study.SPEEDS),seed=8675309,repetitions=50,
                spacing_m=.6,min_spacing_m=.3,warning_s=1.2,calibration_seed=2026,calibration_repetitions=2)
    config.update(updates)
    return config


class ResearchMainTests(unittest.TestCase):
    def test_helpers_do_not_depend_on_fibonacci_runner(self):
        self.assertEqual(study.shared.__name__, "experiments.evaluation_common")
        self.assertEqual(study.shared.run_online.__module__, "experiments.evaluation_common")
        self.assertFalse(hasattr(study.shared, "encounters"))

    @classmethod
    def setUpClass(cls):
        cls.cases=list(study.encounters(configuration()))

    def test_full_design_and_spacing(self):
        self.assertEqual(study.SIZES,(4,6,8,10,16,25))
        self.assertEqual(len(self.cases),3600)
        self.assertEqual(len({c["encounter_id"] for c in self.cases}),3600)
        counts=Counter((c["swarm_size"],c["speed_mps"],c["category"]) for c in self.cases)
        self.assertEqual(set(counts.values()),{50})
        for case in self.cases:
            positions=np.asarray(case["uav_positions"])
            np.testing.assert_allclose(positions.mean(axis=0),0,atol=1e-12)
            d=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
            np.fill_diagonal(d,np.inf)
            self.assertAlmostEqual(float(np.median(d.min(axis=1))),.6)
            self.assertGreaterEqual(float(d.min()),.3)
        means={n:np.mean([c["formation_rms_radius_m"] for c in self.cases if c["swarm_size"]==n]) for n in study.SIZES}
        self.assertGreater(means[25],means[4])

    def test_actual_classes_warning_and_complete_passage(self):
        for case in self.cases:
            p=np.asarray(case["uav_positions"])
            state=np.asarray(case["state"])
            v=state[3:]
            t=np.clip(((p-state[:3])@v)/(v@v),0,case["duration_s"])
            closest=np.linalg.norm(state[:3]+t[:,None]*v-p,axis=1)
            threatened=int(np.sum(closest<.1))
            actual=("multiple_threats" if threatened>=2 else "direct_threat" if np.any(closest<.07)
                    else "grazing_threat" if threatened else "safe_miss")
            self.assertEqual(actual,case["category"])
            if threatened:
                self.assertAlmostEqual(case["first_safety_entry_s"],1.2,places=8)
            else:
                self.assertIsNone(case["first_safety_entry_s"])
            direction=v/np.linalg.norm(v)
            self.assertLess(state[:3]@direction,(p@direction).min()-.1)
            self.assertGreater((state[:3]+v*case["duration_s"])@direction,(p@direction).max()+.55)

    def test_speed_pairs_and_reproducible_internal_layouts(self):
        groups={}
        for case in self.cases:
            positions=np.asarray(case["uav_positions"])
            v=np.asarray(case["state"])[3:]/case["speed_mps"]
            p0=np.asarray(case["state"])[:3]
            line=p0-(p0@v)*v
            if case["geometry_id"] in groups:
                earlier=groups[case["geometry_id"]]
                np.testing.assert_array_equal(positions,earlier[0])
                np.testing.assert_allclose(v,earlier[1],atol=1e-12)
                np.testing.assert_allclose(line,earlier[2],atol=1e-12)
            groups[case["geometry_id"]]=(positions,v,line)
        self.assertEqual(len(groups),1200)
        first=list(study.encounters(configuration(sizes=[4],speeds=[3.0],repetitions=1)))
        self.assertNotEqual(first[0]["uav_positions"],first[1]["uav_positions"])
        self.assertEqual(first,list(study.encounters(configuration(sizes=[4],speeds=[3.0],repetitions=1))))

    def test_independent_calibration_and_threshold_objective(self):
        config=configuration(sizes=[4],speeds=[3.0],repetitions=1,seed=11,calibration_seed=12,calibration_repetitions=1)
        threshold,training,scores,details=study.calibrate(config)
        self.assertTrue(np.isfinite(threshold))
        self.assertFalse({c["encounter_id"] for c in training}&{c["encounter_id"] for c in study.encounters(config)})
        truth=np.array([r["threatened"] for r in scores])
        hits=np.array([r["score"]<=threshold for r in scores])
        balanced=(hits[truth].mean()+(~hits[~truth]).mean())/2
        self.assertAlmostEqual(balanced,details["balanced_accuracy"])
        self.assertTrue(np.all(hits[truth]))
        self.assertEqual(len(scores),16)

    def test_worker_routes_all_four_controllers(self):
        from experiments import online_avoidance
        from baselines import dopplerdodge,dopplerdodge_coordinated
        case=self.cases[0]
        result=dict(status="ok",safety_satisfied=True)
        for method in study.METHODS[1:]:
            received=[]
            connection=SimpleNamespace(send=lambda r:received.append(r),close=lambda:None)
            with patch.object(study.shared,"run_online",return_value=result) as online, \
                 patch.object(study.shared,"run_dopplerdodge",return_value=result) as dodge:
                study.worker(connection,case,method,10,7)
            self.assertEqual(received,[result])
            if method in ("oracle","estimate"):
                self.assertEqual(online.call_args.args[1],method)
                self.assertEqual(dodge.call_count,0)
            else:
                self.assertEqual(online.call_count,0)
                self.assertEqual(dodge.call_args.kwargs.get("coordinated",False),method=="dopplerdodge_coordinated")

    def test_process_timeout_unknown_not_collision(self):
        row=study.run_bounded(self.cases[0],"oracle",10,7,.001)
        self.assertEqual(row["status"],"timeout")
        self.assertIsNone(row["safety_satisfied"])
        self.assertNotIn("projectile_collision",row)

    def test_cli_checkpoint_resume_and_summary_with_timeout(self):
        with tempfile.TemporaryDirectory(prefix="research_main_check_") as temporary:
            output=Path(temporary)/"study"
            args=["research_main","--output",str(output),"--swarm-sizes","4","--speeds","3",
                  "--repetitions","1","--seed","11","--calibration-seed","12","--dodge-threshold","10"]
            def controller(case,method,*args):
                if method=="estimate":
                    return dict(status="timeout",safety_satisfied=None,error="development stub")
                return dict(study.shared.static_result(case),initialization_success=True,
                            partial_timing_screen_met=False,coordinated_plan_success=True)
            with patch.object(sys,"argv",args),patch.object(study,"run_bounded",side_effect=controller), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            rows=study.shared.load_jsonl(output/"results.jsonl")
            self.assertEqual(len(rows),20)
            self.assertEqual({r["method"] for r in rows},set(study.METHODS))
            self.assertTrue((output/"source.zip").exists())
            with (output/"summary.csv").open(newline="") as file:
                summaries=list(csv.DictReader(file))
            estimate=next(s for s in summaries if s["method"]=="estimate")
            self.assertEqual(estimate["observed_safety_rate"],"")
            self.assertEqual(estimate["projectile_collision_rate"],"")
            self.assertEqual(estimate["timeouts"],"1")
            oracle=next(s for s in summaries if s["method"]=="oracle")
            self.assertEqual(float(oracle["initialization_success_p95"]),1.0)
            self.assertEqual(float(oracle["partial_timing_screen_met_p95"]),0.0)
            self.assertEqual(float(oracle["coordinated_plan_success_mean"]),1.0)
            with patch.object(sys,"argv",args+["--resume"]), \
                 patch.object(study,"run_bounded",side_effect=AssertionError("reran completed trial")), \
                 patch.object(study,"calibrate",side_effect=AssertionError("recalibrated")), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            self.assertEqual(len(study.shared.load_jsonl(output/"results.jsonl")),20)

            refreshed=Path(temporary)/"refreshed"
            original_bytes=(output/"results.jsonl").read_bytes()
            refresh_args=["research_main","--output",str(refreshed),"--reuse-results",str(output)]
            training=list(study.encounters(configuration(sizes=[4],speeds=[3.],seed=12,repetitions=1)))
            with patch.object(sys,"argv",refresh_args), \
                 patch.object(study,"calibrate",return_value=(7.,training,[],{"hit_recall":1.})) as calibration, \
                 patch.object(study,"run_bounded",side_effect=controller) as run, \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            self.assertEqual(run.call_count,8)
            self.assertEqual({call.args[1] for call in run.call_args_list},
                             {"dopplerdodge","dopplerdodge_coordinated"})
            self.assertEqual(calibration.call_args.args[0]["sizes"],[4])
            refreshed_rows=study.shared.load_jsonl(refreshed/"results.jsonl")
            retained=lambda rs:[r for r in rs if r["method"] in ("static","oracle","estimate")]
            self.assertEqual(retained(refreshed_rows),retained(rows))
            self.assertEqual((output/"results.jsonl").read_bytes(),original_bytes)
            self.assertEqual((output/"encounters.jsonl").read_bytes(),(refreshed/"encounters.jsonl").read_bytes())
            manifest=json.loads((refreshed/"manifest.json").read_text())
            self.assertEqual(manifest["dopplerdodge_threshold"],7.)
            self.assertIn("estimate",manifest["method_provenance"])
            self.assertTrue((refreshed/"inherited_source.zip").exists())
            with patch.object(sys,"argv",["research_main","--output",str(refreshed),"--resume"]), \
                 patch.object(study,"run_bounded",side_effect=AssertionError("reran")), \
                 patch.object(study,"calibrate",side_effect=AssertionError("recalibrated")), \
                 patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.main_cli()
            changed=study.fingerprint()
            changed["source_hashes"]=dict(changed["source_hashes"])
            changed["source_hashes"][str(Path("estimation")/"ekf.py")]="changed"
            with self.assertRaisesRegex(ValueError,"unrelated source"):
                study.prepare_dodge_refresh(output,Path(temporary)/"invalid",changed)


if __name__=="__main__":
    unittest.main()
