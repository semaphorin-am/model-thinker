"""Tests for evaluation metrics."""

import numpy as np
import pytest

from congestion_pricing.evaluation.metrics import (
    geh_statistic,
    geh_pass_rate,
    rmse,
    mape,
    r_squared,
    mae,
)


class TestGEHStatistic:
    """Tests for GEH statistic."""

    def test_perfect_match(self):
        """GEH should be 0 for identical values."""
        modeled = np.array([100, 200, 300])
        observed = np.array([100, 200, 300])

        geh = geh_statistic(modeled, observed)

        np.testing.assert_array_almost_equal(geh, [0, 0, 0])

    def test_typical_values(self):
        """Test GEH for typical traffic counts."""
        modeled = np.array([1000])
        observed = np.array([1050])

        geh = geh_statistic(modeled, observed)

        # GEH = sqrt(2 * (1000-1050)^2 / (1000+1050))
        expected = np.sqrt(2 * 50**2 / 2050)
        np.testing.assert_almost_equal(geh[0], expected, decimal=4)

    def test_geh_threshold(self):
        """Test GEH < 5 is considered good."""
        # For GEH < 5, |M-O| should be less than about 5% for large flows
        modeled = np.array([1000])
        observed = np.array([1040])

        geh = geh_statistic(modeled, observed)
        assert geh[0] < 5.0

    def test_zero_values(self):
        """Test handling of zero values."""
        modeled = np.array([0, 100])
        observed = np.array([0, 100])

        geh = geh_statistic(modeled, observed)
        # Should not produce NaN or inf
        assert np.all(np.isfinite(geh))


class TestGEHPassRate:
    """Tests for GEH pass rate."""

    def test_all_pass(self, sample_flows):
        """Test when all links pass."""
        modeled, observed = sample_flows
        # Make them very similar
        modeled_close = observed * 1.01

        rate = geh_pass_rate(modeled_close, observed, threshold=10.0)
        assert rate > 0.9

    def test_threshold_effect(self):
        """Test different thresholds."""
        modeled = np.array([100, 200, 150])
        observed = np.array([110, 180, 155])

        rate_5 = geh_pass_rate(modeled, observed, threshold=5.0)
        rate_10 = geh_pass_rate(modeled, observed, threshold=10.0)

        assert rate_10 >= rate_5


class TestRMSE:
    """Tests for RMSE."""

    def test_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])

        assert rmse(predicted, actual) == 0.0

    def test_known_value(self):
        """Test with known RMSE value."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([2.0, 3.0, 4.0])

        # All errors are 1.0, so RMSE = 1.0
        assert rmse(predicted, actual) == 1.0

    def test_typical_values(self, sample_travel_times):
        """Test with typical travel times."""
        predicted, observed = sample_travel_times
        error = rmse(predicted, observed)

        # Should be positive and reasonable
        assert error > 0
        assert error < 10  # Error should be less than 10 minutes


class TestMAPE:
    """Tests for MAPE."""

    def test_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        predicted = np.array([10.0, 20.0, 30.0])
        actual = np.array([10.0, 20.0, 30.0])

        assert mape(predicted, actual) == 0.0

    def test_known_value(self):
        """Test with known MAPE value."""
        predicted = np.array([11.0, 22.0])
        actual = np.array([10.0, 20.0])

        # 10% error for both
        np.testing.assert_almost_equal(mape(predicted, actual), 0.1)

    def test_zero_actual(self):
        """Test handling of zero actual values."""
        predicted = np.array([10.0, 0.0])
        actual = np.array([0.0, 10.0])

        # Should skip zero actual values
        result = mape(predicted, actual)
        assert np.isfinite(result)


class TestRSquared:
    """Tests for R-squared."""

    def test_perfect_prediction(self):
        """R-squared should be 1 for perfect predictions."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        np.testing.assert_almost_equal(r_squared(predicted, actual), 1.0)

    def test_no_correlation(self):
        """R-squared should be low for random predictions."""
        np.random.seed(42)
        predicted = np.random.randn(100)
        actual = np.random.randn(100)

        r2 = r_squared(predicted, actual)
        assert r2 < 0.1

    def test_negative_r_squared(self):
        """R-squared can be negative for very poor fits."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([10.0, 20.0, 30.0])

        r2 = r_squared(predicted, actual)
        assert r2 < 0  # Worse than mean prediction


class TestMAE:
    """Tests for MAE."""

    def test_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])

        assert mae(predicted, actual) == 0.0

    def test_known_value(self):
        """Test with known MAE value."""
        predicted = np.array([1.0, 3.0, 5.0])
        actual = np.array([2.0, 3.0, 4.0])

        # Errors: 1, 0, 1 -> MAE = 2/3
        np.testing.assert_almost_equal(mae(predicted, actual), 2/3)
