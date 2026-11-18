"""Regression checks for startup occupancy and causal audio boundary artifacts."""

import unittest

import numpy as np

from baselines import dopplerdodge as dodge
from simulation.acoustic_sensor import SAMPLE_RATE, _echo_from_history


class DopplerDodgeTests(unittest.TestCase):
    def test_partial_onset_is_excluded_from_drift(self):
        size = 960
        for onset in (1, 300, 700, 950):
            with self.subTest(onset=onset):
                t = np.arange(size * 12) / SAMPLE_RATE
                echo = .5 * np.sin(2 * np.pi * 18123 * t + .3)
                echo[:onset] = 0
                result = dodge.analyze_echo(echo)
                # Independent Section 2.5 reference using only complete buffers.
                buffers = echo.reshape(-1, size)
                rms = np.sqrt(np.mean(buffers ** 2, axis=1))
                first = np.flatnonzero(rms >= dodge.MIN_RMS)[0] + 1
                sums = np.array([np.dot(a, b) for a, b in zip(buffers[first:-1], buffers[first+1:])])
                np.testing.assert_allclose(result.product_sums, sums, atol=1e-12)
                self.assertAlmostEqual(result.score, np.abs(np.diff(sums)).sum())
                self.assertLess(result.score, 10)

    def test_streaming_and_complete_echo_agree_at_chunk_boundaries(self):
        physics = np.arange(49) / 240
        times = np.arange(19200) / SAMPLE_RATE
        for speed in (1, 3, 5):
            path = np.column_stack((speed * (1.2 - physics), physics * 0, physics * 0))
            sensors = np.zeros_like(path)
            complete = dodge.dodge_echo_from_history(times, physics, path, sensors)
            pieces = []
            previous = 0
            for index in range(1, len(physics)):
                end = min(index * 400 + 1, len(times))
                pieces.append(dodge.dodge_echo_from_history(
                    times[previous:end], physics[:index+1], path[:index+1], sensors[:index+1]))
                previous = end
            streaming = np.concatenate(pieces)
            np.testing.assert_allclose(streaming, complete, atol=1e-9, rtol=0)
            self.assertAlmostEqual(dodge.analyze_echo(streaming).score, dodge.analyze_echo(complete).score, places=7)
            # The tolerance must still reject genuinely future receive times.
            future = dodge.dodge_echo_from_history(np.array([.201]), physics, path, sensors)
            np.testing.assert_array_equal(future, 0)

    def test_recall_first_threshold_and_unobservable_hit(self):
        scores = np.array([2., 5., 3., 10.])
        truth = np.array([True, True, False, False])
        threshold = dodge.select_hit_threshold(scores, truth)
        self.assertTrue(np.all(scores[truth] <= threshold))
        self.assertLess(threshold, 10)
        with self.assertRaisesRegex(ValueError, "no usable echo"):
            dodge.select_hit_threshold([np.inf, 1], [True, False])

    def test_shared_sensor_defaults_preserve_existing_estimator_audio(self):
        physics = np.arange(49) / 240
        path = np.column_stack((1.2 - physics, physics * 0, physics * 0))
        sensors = np.zeros_like(path)
        times = np.arange(19200) / SAMPLE_RATE
        default = _echo_from_history(times, physics, path, sensors)
        explicit = _echo_from_history(times, physics, path, sensors,
                                      carrier_frequency=10000., time_tolerance=0.)
        np.testing.assert_array_equal(default, explicit)


if __name__ == "__main__":
    unittest.main()
