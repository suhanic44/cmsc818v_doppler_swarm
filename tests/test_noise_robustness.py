"""Verify noise-study controls and analysis, without running a research dataset.

python -m unittest discover -s tests -p test_noise_robustness.py
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments import noise_robustness as study
from experiments.fibonacci_evaluation import encounters as fibonacci_encounters


class NoiseTests(unittest.TestCase):
    def test_zero_noise_is_exact(self):
        values = np.array([1.25, -4.0])
        noise = study.MeasurementNoise("case", 7, 0, 0)
        self.assertIs(noise.apply(0, [.04, .06], values), values)
        self.assertEqual(noise.count, 0)

    def test_reproducible_paired_and_batch_independent(self):
        times = np.array([.04,.06,.08,.10])
        noise = lambda sigma, realization=0: study.MeasurementNoise("case",7,realization,sigma)
        full = noise(2).apply(0,times,np.zeros(4))
        split = noise(2)
        batches = np.r_[split.apply(0,times[:2],np.zeros(2)),split.apply(0,times[2:],np.zeros(2))]
        np.testing.assert_array_equal(full,batches)
        np.testing.assert_array_equal(full,2*noise(1).apply(0,times,np.zeros(4)))
        self.assertFalse(np.array_equal(full,noise(2,1).apply(0,times,np.zeros(4))))
        self.assertFalse(np.array_equal(full,noise(2).apply(1,times,np.zeros(4))))

    def test_noise_distribution(self):
        noise = study.MeasurementNoise("distribution",123,0,2)
        values = noise.apply(0,np.arange(3000)*.02,np.zeros(3000))
        self.assertLess(abs(values.mean()),.15)
        self.assertLess(abs(values.std()-2),.15)
        self.assertAlmostEqual(noise.metrics()["injected_rms_hz"],np.sqrt(np.mean(values**2)))

    def test_checkpoint_uses_availability_not_state_timestamp(self):
        case = dict(state=[0,0,0,1,0,0])
        trace = [dict(available_s=.5,state_time_s=.2,state=[.2,0,0,1,0,0])]
        scores = study.checkpoint_metrics(case,trace,.7)
        self.assertEqual(scores[0]["status"],"no_estimate")
        self.assertEqual(scores[1]["status"],"valid")
        self.assertAlmostEqual(scores[1]["forecast_error_m"],0)
        self.assertEqual(scores[2]["status"],"not_reached")

    def test_encounter_clustering_does_not_count_seeds_as_geometries(self):
        rows = [dict(encounter_id="a",safe=v) for v in [0,1,0,1]]
        rows += [dict(encounter_id="b",safe=1)]
        means = study.clustered_metric(rows,lambda r:r["safe"])
        self.assertEqual(means,[.5,1])
        first = study.bootstrap(means,"fixed")
        self.assertEqual(first,study.bootstrap(means,"fixed"))
        self.assertEqual(first[0],.75)

    def test_constant_outcomes_do_not_claim_certain_reliability(self):
        mean,low,high = study.bootstrap([1.0]*50,"success")
        self.assertEqual(mean,1)
        self.assertLess(low,1)
        self.assertEqual(high,1)
        mean,low,high = study.bootstrap([0.0]*50,"paired",bounds=(-1,1))
        self.assertLess(low,0)
        self.assertGreater(high,0)

    def test_trial_plan_unique_and_clean_control_not_duplicated(self):
        case=dict(encounter_id="case")
        tasks=list(study.trials([case],5,7))
        self.assertEqual(len(tasks),53)
        ids=[study.trial_id(c["encounter_id"],method,sigma,policy,seed)
             for c,method,sigma,policy,seed in tasks]
        self.assertEqual(len(set(ids)),53)
        self.assertEqual(sum(task[1]=="estimate" and task[2]==0 for task in tasks),1)

    def test_individual_policies_have_matching_noise_trial_identities(self):
        case=dict(encounter_id="case")
        plans={policy:list(study.trials([case],5,7,(policy,))) for policy in study.POLICIES}
        for policy,tasks in plans.items():
            self.assertEqual(len(tasks),28)
            self.assertEqual({task[3] for task in tasks if task[2]}, {policy})
        identities=lambda tasks:{(c["encounter_id"],sigma,seed) for c,method,sigma,policy,seed in tasks}
        self.assertEqual(identities(plans["fixed"]),identities(plans["variance_aware"]))

    def test_noise_reaches_initialization_and_patches_restore(self):
        class Pipeline:
            def __init__(self):
                self.ekf = SimpleNamespace(sigma_doppler=1.0)
                self.initialized = False
                self.estimated_times = []
                self.estimated_states = []

            def add_measurements(self,sensor,times,values):
                self.values = values

        def process(pipeline,t):
            self.assertFalse(np.array_equal(pipeline.values,np.array([0.0])))
            pipeline.initialized=True
            pipeline.estimated_times.append(t)
            pipeline.estimated_states.append(np.array([t,0,0,1,0,0]))

        online = SimpleNamespace(OnlineEstimationPipeline=Pipeline,process_pipeline_measurements=process)
        init, add = Pipeline.__init__,Pipeline.add_measurements
        def run(case,method,online):
            pipeline = online.OnlineEstimationPipeline()
            self.assertAlmostEqual(pipeline.ekf.sigma_doppler,np.sqrt(5))
            pipeline.add_measurements(0,[.04],np.array([0.0]))
            online.process_pipeline_measurements(pipeline,.3)
            return dict(status="ok",safety_satisfied=True)
        case = dict(encounter_id="case",state=[0,0,0,1,0,0])
        with patch.object(study.main,"run_online",run):
            result = study.run_estimate(case,2,"variance_aware",0,7,online)
        self.assertTrue(result["initialization_success"])
        self.assertEqual(result["injected_samples"],1)
        self.assertIs(Pipeline.__init__,init)
        self.assertIs(Pipeline.add_measurements,add)
        self.assertIs(online.process_pipeline_measurements,process)
        with patch.object(study.main,"run_online",side_effect=RuntimeError("initialization failed")):
            failure = study.run_estimate(case,2,"fixed",0,7,online)
        self.assertEqual(failure["status"],"error")
        self.assertFalse(failure["initialization_success"])
        self.assertIs(Pipeline.__init__,init)

    def test_cli_runs_only_selected_policies_for_all_sizes(self):
        with tempfile.TemporaryDirectory(prefix="noise_cli_check_") as temporary:
            root=Path(temporary)
            source=root/"source"
            source.mkdir()
            study.write_jsonl(source/"encounters.jsonl",list(fibonacci_encounters(314159,1)))
            study.write_json(source/"manifest.json",dict(
                source_hashes=study.main.source_hashes(),environment=study.fingerprint()["environment"],
                encounters_sha256=study.hashlib.sha256((source/"encounters.jsonl").read_bytes()).hexdigest()))
            def estimate(case,*args):
                row=study.main.static_result(case)
                row.update(checkpoint_scores=study.checkpoint_metrics(case,[],case["duration_s"]))
                return row
            for selection in ("fixed","variance_aware","both"):
                output=root/selection
                arguments=["experiments.noise_robustness","--source",str(source),"--output",str(output),"--realizations","1"]
                if selection!="fixed":  # Also verify the default really runs fixed only.
                    arguments += ["--filter-policy",selection]
                with patch.object(sys,"argv",arguments),patch.object(study,"run_estimate",side_effect=estimate), \
                     patch.object(study.main,"run_online",side_effect=lambda c,*args:study.main.static_result(c)), \
                     patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                    study.main_cli()
                rows=study.main.load_jsonl(output/"results.jsonl")
                expected=set(study.POLICIES) if selection=="both" else {selection}
                self.assertEqual({r["filter_policy"] for r in rows if r["sigma_hz"]},expected)
                self.assertEqual(len(rows),16*(3+5*len(expected)))
                manifest=json.loads((output/"manifest.json").read_text())
                self.assertEqual(manifest["config"]["sizes"],[4,6,8,10])

    def test_study_snapshots_and_resume_and_analysis(self):
        with tempfile.TemporaryDirectory(prefix="noise_study_check_") as temporary:
            root = Path(temporary)
            source = root/"source"
            source.mkdir()
            cases = list(fibonacci_encounters(314159,2))
            study.write_jsonl(source/"encounters.jsonl",cases)
            parent = dict(source_hashes=study.main.source_hashes(),environment=study.fingerprint()["environment"],
                          encounters_sha256=study.hashlib.sha256((source/"encounters.jsonl").read_bytes()).hexdigest())
            study.write_json(source/"manifest.json",parent)
            # A concurrent writer's incomplete line must not be treated as a completed result.
            (source/"results.jsonl").write_text('{"encounter_id":',encoding="utf-8")
            args = SimpleNamespace(source=source,output=root/"study",resume=False,seed=7,realizations=2,filter_policy="both")
            selected, manifest = study.initialize_study(args)
            self.assertEqual(len(selected),32)
            self.assertEqual({c["swarm_size"] for c in selected},{4,6,8,10})
            self.assertEqual(manifest["planned_trials"],32*23)
            args.resume=True
            resumed, second = study.initialize_study(args)
            self.assertEqual(selected,resumed)
            self.assertEqual(manifest,second)
            args.filter_policy="fixed"
            with self.assertRaisesRegex(ValueError,"configuration"):
                study.initialize_study(args)
            args.filter_policy="both"
            for policy in study.POLICIES:
                separate=SimpleNamespace(source=source,output=root/policy,resume=False,seed=7,realizations=2,filter_policy=policy)
                individual,metadata=study.initialize_study(separate)
                self.assertEqual(individual,selected)
                self.assertEqual(metadata["planned_trials"],32*13)
                self.assertEqual(metadata["config"]["policies"],[policy])
            chosen = [c for c in selected if c["swarm_size"]==4 and c["category"]=="direct_threat"]
            results=[]
            for case in chosen:
                for sigma,policy,seeds in ((0.0,"baseline",[0]),(2.0,"fixed",[0,1])):
                    for realization in seeds:
                        row = dict(trial_id=study.trial_id(case["encounter_id"],"estimate",sigma,policy,realization),
                                   encounter_id=case["encounter_id"],method="estimate",sigma_hz=sigma,
                                   filter_policy=policy,noise_realization=realization,
                                   status="ok",safety_satisfied=(sigma==0 or realization==0),
                                   path_length_m=.1,initialization_success=True,assumed_sigma_hz=1.0,
                                   checkpoint_scores=study.checkpoint_metrics(case,[],case["duration_s"]))
                        results.append(row)
            study.write_jsonl(args.output/"results.jsonl",results)
            with patch.object(study,"plot_results"),contextlib.redirect_stdout(io.StringIO()):
                study.summarize(args.output)
            import csv
            with (args.output/"summary.csv").open(newline="",encoding="utf-8") as file:
                summaries=list(csv.DictReader(file))
            noisy=next(s for s in summaries if s["sigma_hz"]=="2.0")
            self.assertEqual(float(noisy["safety_rate"]),.5)
            self.assertEqual(int(noisy["encounters"]),2)
            self.assertEqual(float(noisy["safety_change_vs_clean"]),-.5)
            self.assertTrue((args.output/"checkpoint_metrics.csv").exists())
            # Changing an immutable input must prevent resume.
            with (args.output/"encounters.jsonl").open("a",encoding="utf-8") as file:
                file.write("\n")
            with self.assertRaisesRegex(ValueError,"Saved input changed"):
                study.initialize_study(args)


if __name__ == "__main__":
    unittest.main()
