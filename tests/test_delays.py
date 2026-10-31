"""Alignment and HCI integration checks, runnable without IDTxl/JIDT."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from joblib import parallel_backend


class GaussianMI:
    """Deterministic stand-in; does not validate JIDT numerical estimates."""
    def __init__(self, settings):
        pass

    def estimate(self, x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        rho = np.corrcoef(x.ravel(), y.ravel())[0, 1]
        return -0.5 * np.log(max(1.0 - rho * rho, 1e-12))


# Isolate the stand-in so importing these tests does not replace real IDTxl.
stub = types.ModuleType("idtxl.estimators_jidt")
stub.JidtKraskovMI = GaussianMI
spec = importlib.util.spec_from_file_location(
    "hci_under_test", Path(__file__).resolve().parents[1] / "hci_v3.py"
)
hci = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"idtxl": types.ModuleType("idtxl"),
                             "idtxl.estimators_jidt": stub}):
    spec.loader.exec_module(hci)


class DelayTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(80).reshape(20, 4)

    def test_positive_negative_and_common_offset(self):
        expected = np.column_stack([self.X[:-5, 0], self.X[5:, 1],
                                    self.X[3:-2, 2], self.X[4:-1, 3]])
        for delays in ([5, 0, 2, 1], [0, -5, -3, -4], [105, 100, 102, 101]):
            np.testing.assert_array_equal(hci.align_signals(self.X, delays), expected)

    def test_no_delay_and_equal_delays_preserve_input(self):
        for delays in (None, [0]*4, [-9]*4):
            np.testing.assert_array_equal(hci.align_signals(self.X, delays), self.X)

    def test_invalid_delays(self):
        for delays in ([0, 1], [[0]*4], [0, 0, 0, 1.5], [0, 0, 0, True],
                       [0, 0, 0, np.nan], [0, 0, 0, "1"], [0, 0, 0, 20]):
            with self.subTest(delays=delays), self.assertRaises(ValueError):
                hci.align_signals(self.X, delays)
        with self.assertRaises(ValueError):
            hci.align_signals(np.arange(20), [0])

    def test_single_sample_overlap_and_no_mutation(self):
        original = self.X.copy()
        actual = hci.align_signals(self.X, [19, 0, 0, 0])
        np.testing.assert_array_equal(actual, [[0, 77, 78, 79]])
        np.testing.assert_array_equal(self.X, original)

    def test_delayed_copy_alignment(self):
        rng = np.random.default_rng(123)
        audio = rng.normal(size=1000)
        eeg = np.r_[rng.normal(size=14), audio[:-14]]
        aligned = hci.align_signals(np.column_stack([audio, eeg]), [14, 0])
        np.testing.assert_array_equal(aligned[:, 0], aligned[:, 1])

    def test_public_estimators_match_manual_alignment(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 4)) + rng.normal(size=(100, 1))
        # Explicit slicing independently checks the sign convention.
        manual = np.column_stack([X[:-3, 0], X[3:, 1], X[2:-1, 2], X[1:-2, 3]])
        delays = [3, 0, 1, 2]
        kwargs = dict(repeats=2, max_itr=3)
        calls = [(hci.compute_R2_R3, (), dict(target_idx=2, x1_idx=0, x2_idx=1)),
                 (hci.R3_approx_mi_fast, (2, 0, 1), {}),
                 (hci.R4_approx_mi_fast, (2, 0, 3, 1), dict(refine_repeats=2))]
        for func, args, extra in calls:
            with self.subTest(function=func.__name__):
                expected = func(manual, *args, **kwargs, **extra)
                np.testing.assert_allclose(
                    func(X, *args, delays=delays, **kwargs, **extra), expected)
                np.testing.assert_allclose(
                    func(X, *args, delays=[0]*4, **kwargs, **extra),
                    func(X, *args, **kwargs, **extra))

    def test_parallel_dispatch_matches_serial(self):
        X = np.random.default_rng(5).normal(size=(100, 3))
        kwargs = dict(repeats=2, max_itr=3, delays=[2, 0, 1])
        expected = hci.R3_approx_mi_fast(X, 0, 1, 2, **kwargs)
        # Threads exercise the parallel branch without a JVM in worker processes.
        with parallel_backend("threading"):
            actual = hci.R3_approx_mi_fast(X, 0, 1, 2, n_jobs=2, **kwargs)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
