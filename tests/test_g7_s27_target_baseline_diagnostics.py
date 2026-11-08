"""Synthetic-only target origin controls; never opens S25/S26 scientific worlds."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from g7_s27_target_baseline_diagnostics import (
    DiagnosticInvalid,GAMMAS,RIDGES,TargetAdjustedKRR,
    five_check_spread,numeric_vectors,synthetic_paired_world)

class TestG7S27TargetOrigin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train=synthetic_paired_world(16)
        cls.val=synthetic_paired_world(8)

    def test_grid_and_mode_frozen(self):
        self.assertEqual(GAMMAS,(.5,2.,6.))
        self.assertEqual(RIDGES,(.01,.08))
        for m in ('raw','new',None):
            with self.assertRaisesRegex(DiagnosticInvalid,'UNREGISTERED'):
                TargetAdjustedKRR(m)

    def test_exact_131_features_and_shape(self):
        h,q,_=self.train
        self.assertEqual(tuple(numeric_vectors(h,q).shape),(32,131))
        m=TargetAdjustedKRR('train_mean_centered')
        m.fit_calibrated(self.train,self.val)
        self.assertEqual(tuple(m.predict(self.val[0],self.val[1]).shape),(8,2))

    def test_untrained_candidate_is_blocked(self):
        with self.assertRaisesRegex(DiagnosticInvalid,'UNTRAINED'):
            TargetAdjustedKRR('current_sensor_residual').predict(self.val[0],self.val[1])

    def test_train_mean_centering_is_target_offset_equivariant(self):
        a=TargetAdjustedKRR('train_mean_centered')
        a.fit_calibrated(self.train,self.val)
        initial=a.predict(self.val[0],self.val[1])
        shift=lambda x:(x[0],x[1],x[2]+4.25)
        b=TargetAdjustedKRR('train_mean_centered')
        b.fit_calibrated(shift(self.train),shift(self.val))
        self.assertTrue(torch.allclose(b.predict(self.val[0],self.val[1]),
                                       initial+4.25,rtol=0,atol=2e-5))
        self.assertTrue(torch.equal(a.mean,b.mean))
        self.assertEqual((a.gamma,a.ridge),(b.gamma,b.ridge))

    def test_current_sensor_residual_is_joint_offset_equivariant(self):
        a=TargetAdjustedKRR('current_sensor_residual')
        a.fit_calibrated(self.train,self.val)
        initial=a.predict(self.val[0],self.val[1])
        def shift(d):
            h,q,y=(v.clone() for v in d)
            h[:,0,3]+=3.5
            q[:,:,0]+=3.5
            y+=3.5
            return h,q,y
        tr,val=shift(self.train),shift(self.val)
        b=TargetAdjustedKRR('current_sensor_residual')
        b.fit_calibrated(tr,val)
        self.assertTrue(torch.allclose(b.predict(val[0],val[1]),
                                       initial+3.5,atol=2e-5,rtol=0))
        self.assertEqual((a.gamma,a.ridge),(b.gamma,b.ridge))

    def test_train_only_normalization_ignores_validation_targets(self):
        a=TargetAdjustedKRR('train_mean_centered')
        a.fit_calibrated(self.train,self.val)
        h,q,y=self.val
        b=TargetAdjustedKRR('train_mean_centered')
        b.fit_calibrated(self.train,(h,q,y+91))
        self.assertTrue(torch.equal(a.mean,b.mean))
        self.assertTrue(torch.equal(a.scale,b.scale))
        self.assertTrue(torch.equal(a.target_mean,b.target_mean))

    def test_invalid_shapes_or_nonfinite_fail(self):
        h,q,y=self.train
        for badq in (q[:,:1,:],q[:,:,:2]):
            with self.assertRaises(DiagnosticInvalid):
                numeric_vectors(h,badq)
        bad=h.clone();bad[0,0,0]=float('nan')
        with self.assertRaisesRegex(DiagnosticInvalid,'NONFINITE'):
            numeric_vectors(bad,q)
        bady=y.clone();bady[0,0]=float('inf')
        with self.assertRaises(DiagnosticInvalid):
            TargetAdjustedKRR('train_mean_centered').fit_calibrated((h,q,bady),self.val)

    def test_pair_order_and_common_state_cannot_drift(self):
        h,q,_=self.train
        for index,value in ((0,.1),(1,.9)):
            bad=q.clone()
            if index==0:bad[:,1,0]+=.1
            else:bad[:,1,1]=value
            with self.assertRaisesRegex(DiagnosticInvalid,'ACTION_PROBE'):
                numeric_vectors(h,bad)

    def test_gru_plateau_preserves_old_strict_comparison(self):
        self.assertEqual(len(five_check_spread((.1,.1,.1,.1,.1,.2))),2)
        self.assertEqual(five_check_spread((1.,1.,1.,1.,1.)),(0.,))
        self.assertFalse(five_check_spread((1.,1.,1.,1.,1.0011))[0]<.001)
        with self.assertRaises(DiagnosticInvalid):
            five_check_spread((0.,1.,1.,1.,1.))

    def test_no_registered_science_world_or_fitter_import(self):
        src=(Path(__file__).resolve().parents[1]/"scripts/g7_s27_target_baseline_diagnostics.py").read_text()
        for forbidden in ('PairedS20World','registered_role=','7207','7307','92017',
                          'torch.optim','fit_conventional_gru','g7_s26_v2_firstuse_controls'):
            self.assertNotIn(forbidden,src)
        self.assertIn('Invented numeric engineering fixture',src)

if __name__=='__main__':
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
