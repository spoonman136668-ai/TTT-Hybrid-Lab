"""S26-V2 first-use admission negative controls. Run BEFORE any fit/validation seed.

No experiment() call, no training, no fit/validation world instance,
no checkpoint read, and no scientific seed consumption.
"""
from __future__ import annotations
import ast
import json
import math
import sys
from pathlib import Path
import unittest

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_v2_firstuse_controls as science
from g7_s25_paired_interventions import PairedS20World
from g7_s25_numeric_ttt import ConventionalGRU,CalibratedRBFKernelRidge


class TestS26V2FirstUsePreregistered(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.p=json.loads((ROOT/"experiments/g7_s26_v2_fitval_firstuse_frozen.json").read_text())

    def test_original_frozen_protocols_never_modified(self):
        self.assertEqual(science.PROTOCOL_BLOB,"d72b9bb41e5e7e5dd9229c7bc5053f5a42f9750d")
        self.assertEqual(science.PARENT_SHA,"255aa654bd3dfc6d5b36f137555e4d33dcf95cb1")
        self.assertEqual(self.p["original_s26_protocol_blob"],
                         "34251b2cde6373ed484dbc21e4617c9165047e93")
        self.assertTrue(self.p["authority"][
            "original_s25_and_s26_frozen_scientific_admission_still_blocked"])
        original=json.loads((ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())
        self.assertFalse(original["authority"]["scientific_first_use_authorized"])
        self.assertTrue(original["authority"]["no_scientific_training_or_fit"])

    def test_only_fit_and_validation_roles_registered(self):
        self.assertEqual(self.p["role_seeds"]["fit"],list(science.FIT))
        self.assertEqual(self.p["role_seeds"]["validation"],list(science.VALIDATION))
        self.assertEqual(self.p["role_seeds"]["development"],list(science.DEV))
        self.assertTrue(set(science.FIT).isdisjoint(science.DEV+science.VALIDATION+science.NO_SEEDS))
        self.assertTrue(set(science.VALIDATION).isdisjoint(science.DEV+science.NO_SEEDS))

    def test_non_science_default_world_rejects_all_reserved_seeds(self):
        for seed in (science.FIT+science.VALIDATION+science.NO_SEEDS):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError,"S25_ENGINEERING_SEEDS_ONLY"):
                    PairedS20World(seed,32,"source_stable")

    def test_explicit_registered_world_refuses_science_roles_other_than_fit_val(self):
        for role in ("heldout","transfer","negative","bootstrap",None,"dev","FIT"):
            if role is None:
                continue
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError,"FIRSTUSE_ROLE_NOT_AUTHORIZED"):
                    PairedS20World(7207,32,"source_stable",registered_role=role)

    def test_wrong_fit_validation_seed_or_cross_role_is_rejected_without_consuming(self):
        for seed,role in ((7207,"validation"),(7307,"fit"),
                          (7403,"fit"),(7507,"validation"),
                          (3101,"fit"),(7807,"validation"),(9999,"fit")):
            with self.subTest(seed=seed,role=role):
                with self.assertRaisesRegex(ValueError,"FIRSTUSE_SEED_OR_AUTHORITY_MISMATCH"):
                    PairedS20World(seed,32,"source_stable",registered_role=role)

    def test_precommitted_gru_trainer_and_krr_controls_exist_unmodified(self):
        self.assertTrue(callable(ConventionalGRU.predict))
        self.assertTrue(callable(science.numeric.fit_conventional_gru))
        self.assertEqual(CalibratedRBFKernelRidge.GAMMAS,(0.5,2.0,6.0))
        self.assertEqual(CalibratedRBFKernelRidge.RIDGES,(0.01,0.08))
        self.assertEqual(self.p["controls"]["GRU_fully_trained"]["max_updates"],1600)
        self.assertEqual(self.p["world"]["selected_samples_per_role_seed"],128)

    def test_static_synthetic_good_mse_but_wrong_sign_fails_criteria(self):
        h=torch.zeros(4,32,4)
        q=torch.zeros(4,2,3)
        q[:,:,1]=torch.tensor([-.8,.8])
        y=torch.tensor([[0.,.1]]*4)
        a=torch.zeros(4,dtype=torch.long)
        data=science.SourceBatch(h,q,y,a,"a"*64,"b"*64,7207,"fit",1024,128)
        # Good MSE relative to a deliberately terrible static baseline,
        # but wrong sign. Must be NOT admitted without any model training.
        pred=torch.tensor([[.09,.01]]*4)
        score=science.validation_readout(pred,data)
        self.assertFalse(score["precommitted_admission_criteria_met"])
        self.assertEqual(score["action_effect_correct_sign"],0)
        self.assertEqual(score["action_effect_wrong_sign"],4)

    def test_synthetic_near_zero_effect_is_uninformative(self):
        h=torch.zeros(4,32,4)
        q=torch.zeros(4,2,3)
        y=torch.tensor([[0.,.1]]*4)
        data=science.SourceBatch(h,q,y,torch.zeros(4,dtype=torch.long),
                                 "a"*64,"b"*64,7207,"fit",1024,128)
        score=science.validation_readout(torch.tensor([[.049,.051]]*4),data)
        self.assertEqual(score["action_effect_eligible_predicted"],0)
        self.assertFalse(score["precommitted_admission_criteria_met"])

    def test_nan_predictions_fail_closed(self):
        h=torch.zeros(4,32,4);q=torch.zeros(4,2,3);y=torch.zeros(4,2)
        data=science.SourceBatch(h,q,y,torch.zeros(4,dtype=torch.long),
                                 "a"*64,"b"*64,7207,"fit",1024,128)
        with self.assertRaisesRegex(science.FirstUseBlocked,"NONFINITE_VALIDATION"):
            science.validation_readout(torch.full((4,2),float("nan")),data)

    def test_cli_must_be_project_owned_cpu_fit_validation_only(self):
        source=(ROOT/"scripts/g7_s26_v2_firstuse_controls.py").read_text()
        for key in ("GITHUB_REPOSITORY","GITHUB_REF","GITHUB_SHA",
                    "RUNNER_NAME","S26_V2_PROJECT_FIRST_USE",
                    "CUDA_VISIBLE_DEVICES"):
            self.assertIn(key,source)
        self.assertIn("torch.load(path,map_location=\"cpu\",weights_only=True)",
                      source)
        self.assertNotIn("torch.cuda.synchronize",source)

    def test_original_data_generation_mechanism_unmodified_except_role_gate(self):
        path=ROOT/"scripts/g7_s25_paired_interventions.py"
        source=path.read_text()
        tree=ast.parse(source)
        cls=next(n for n in tree.body
                 if isinstance(n,ast.ClassDef) and n.name=="PairedS20World")
        methods=[n.name for n in cls.body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(methods,["__init__","_counterfactual_physics","emit","close"])
        self.assertIn("S25_ENGINEERING_SEEDS_ONLY",source)
        self.assertIn("registered_role",source)
        self.assertIn("FIRSTUSE_SEED_OR_AUTHORITY_MISMATCH",source)
        self.assertIn("source_stable",source)

    def test_true_fit_validation_seeds_not_read_during_any_test(self):
        tree=ast.parse((ROOT/"tests/test_g7_s26_v2_firstuse_contract.py").read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        function_calls={n.func.id for n in calls if isinstance(n.func,ast.Name)}
        self.assertNotIn("experiment",function_calls)
        self.assertNotIn("paired_source",function_calls)
        self.assertNotIn("fit_conventional_gru",function_calls)
        self.assertNotIn("fit_calibrated",function_calls)


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=1))
