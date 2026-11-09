import numpy as np
import pytest

from fraud_api.evaluation import (
    fpr_at,
    fpr_ratio,
    paired_bootstrap,
    pr_auc,
    precision_ratio,
    recall_at,
    recall_at_fpr,
    threshold_for_fpr,
)

# Twenty Legitimate Applications scored 0.01, 0.02, ..., 0.20, and five Fraudulent ones.
LEGITIMATE_SCORES = [i / 100 for i in range(1, 21)]
FRAUDULENT_SCORES = [0.30, 0.195, 0.19, 0.15, 0.10]
LABELS = [0] * 20 + [1] * 5
SCORES = LEGITIMATE_SCORES + FRAUDULENT_SCORES


class TestThreshold:
    def test_flags_exactly_the_target_share_of_legitimate_applications(self):
        # 5% of 20 is one: only the Legitimate Application at 0.20 may be flagged.
        threshold = threshold_for_fpr(LABELS, SCORES, 0.05)

        assert fpr_at(LABELS, SCORES, threshold) == 0.05
        assert 0.19 < threshold <= 0.195

    def test_is_the_lowest_threshold_within_the_target(self):
        threshold = threshold_for_fpr(LABELS, SCORES, 0.05)

        # Dropping to the next Legitimate score would flag two of twenty.
        assert fpr_at(LABELS, SCORES, 0.19) == 0.10
        # Anything between 0.19 and the Threshold keeps a Fraudulent 0.195 flagged.
        assert recall_at(LABELS, SCORES, threshold) == recall_at(LABELS, SCORES, 0.195)

    def test_tie_straddling_the_target_moves_the_threshold_above_the_tie(self):
        # Two Legitimate Applications tie at 0.9; flagging both would be 10% FPR.
        labels = [0] * 20 + [1, 1]
        scores = [i / 100 for i in range(1, 19)] + [0.9, 0.9] + [0.95, 0.9]

        threshold = threshold_for_fpr(labels, scores, 0.05)

        assert fpr_at(labels, scores, threshold) == 0.0
        # The Fraudulent Application tied with them at 0.9 is not flagged either.
        assert recall_at(labels, scores, threshold) == 0.5

    def test_tie_within_the_target_is_flagged_together(self):
        # Two Legitimate Applications tie at 0.9; 10% of 20 allows both.
        labels = [0] * 20 + [1]
        scores = [i / 100 for i in range(1, 19)] + [0.9, 0.9] + [0.5]

        threshold = threshold_for_fpr(labels, scores, 0.10)

        assert fpr_at(labels, scores, threshold) == 0.10
        assert recall_at(labels, scores, threshold) == 1.0

    def test_target_whose_float_product_lands_just_below_an_integer(self):
        # 0.29 * 100 is 28.999999999999996 in floating point; 29 flags are allowed.
        labels = [0] * 100
        scores = [i / 1000 for i in range(100)]

        threshold = threshold_for_fpr(labels, scores, 0.29)

        assert fpr_at(labels, scores, threshold) == 0.29

    def test_target_just_below_a_whole_flag_does_not_round_up(self):
        # 0.0499999999999 of 20 is just under one flag, so none may be flagged.
        labels = [0] * 20
        scores = [i / 100 for i in range(20)]

        threshold = threshold_for_fpr(labels, scores, 0.0499999999999)

        assert fpr_at(labels, scores, threshold) == 0.0

    def test_target_just_below_one_leaves_the_top_score_unflagged(self):
        threshold = threshold_for_fpr([0], [0.5], 1 - 1e-12)

        assert fpr_at([0], [0.5], threshold) == 0.0

    @pytest.mark.parametrize("target", [0, 1, -0.1, 1.5])
    def test_rejects_targets_outside_zero_and_one(self, target):
        with pytest.raises(ValueError):
            threshold_for_fpr(LABELS, SCORES, target)

    def test_rejects_applications_with_no_legitimate_ones(self):
        with pytest.raises(ValueError):
            threshold_for_fpr([1, 1], [0.2, 0.8], 0.05)


class TestRecallAndFpr:
    def test_recall_at_a_threshold(self):
        # Fraudulent scores at or above 0.15: 0.30, 0.195, 0.19, 0.15.
        assert recall_at(LABELS, SCORES, 0.15) == 0.8

    def test_fpr_at_a_threshold(self):
        # Legitimate scores at or above 0.15: 0.15, ..., 0.20, six of twenty.
        assert fpr_at(LABELS, SCORES, 0.15) == pytest.approx(0.30)

    def test_recall_at_5_percent_fpr(self):
        # Only 0.30 and 0.195 clear the Threshold just above 0.19.
        assert recall_at_fpr(LABELS, SCORES) == 0.4

    def test_recall_needs_fraudulent_applications(self):
        with pytest.raises(ValueError):
            recall_at([0, 0], [0.2, 0.8], 0.5)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fpr_at([0, 1, 0], [0.1, 0.2], 0.5)

    def test_rejects_labels_other_than_zero_and_one(self):
        with pytest.raises(ValueError):
            fpr_at([0, 2], [0.1, 0.2], 0.5)


class TestPrAuc:
    def test_average_precision_without_ties(self):
        # Ranked: F (P=1, R=.5), L, F (P=2/3, R=1), L.
        assert pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)

    def test_tied_scores_count_as_one_step(self):
        # At 0.5 one F and one L enter together (P=.5, R=.5), then F at 0.2 (P=2/3, R=1).
        assert pr_auc([1, 0, 1], [0.5, 0.5, 0.2]) == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)

    def test_needs_fraudulent_applications(self):
        with pytest.raises(ValueError):
            pr_auc([0, 0], [0.1, 0.2])


class TestFprRatio:
    def test_under_50_group_with_the_higher_fpr_is_the_denominator(self):
        # Under 50: 2 of 4 Legitimate flagged (0.5). 50 or over: 1 of 4 (0.25).
        # Fraudulent Applications never count, whatever their age.
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        scores = [0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.9, 0.1]
        ages = [20, 30, 40, 40, 50, 60, 70, 90, 20, 80]

        assert fpr_ratio(labels, scores, ages, 0.5) == 0.5

    def test_50_or_over_group_with_the_higher_fpr_is_the_denominator(self):
        # 50 or over: 3 of 4 flagged (0.75). Under 50: 1 of 4 (0.25).
        labels = [0, 0, 0, 0, 0, 0, 0, 0]
        scores = [0.9, 0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.1]
        ages = [20, 30, 40, 40, 50, 60, 70, 90]

        assert fpr_ratio(labels, scores, ages, 0.5) == pytest.approx(1 / 3)

    def test_age_50_belongs_to_the_older_group(self):
        # The first Applicant is flagged. At 49: under 50 is 2 of 3, 50 or over 1 of 3.
        # At 50: under 50 is 1 of 2, 50 or over 2 of 4, so the groups become equal.
        labels = [0, 0, 0, 0, 0, 0]
        scores = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]

        assert fpr_ratio(labels, scores, [49, 30, 60, 30, 60, 70], 0.5) == 0.5
        assert fpr_ratio(labels, scores, [50, 30, 60, 30, 60, 70], 0.5) == 1.0

    def test_equal_fpr_gives_one(self):
        labels = [0, 0, 0, 0]
        scores = [0.9, 0.1, 0.9, 0.1]
        ages = [20, 30, 60, 70]

        assert fpr_ratio(labels, scores, ages, 0.5) == 1.0

    def test_nobody_flagged_counts_as_equal(self):
        assert fpr_ratio([0, 0], [0.1, 0.2], [20, 60], 0.5) == 1.0

    def test_rejects_missing_ages(self):
        with pytest.raises(ValueError):
            fpr_ratio([0, 0, 0], [0.9, 0.1, 0.9], [float("nan"), 20, 60], 0.5)

    def test_needs_legitimate_applications_in_both_age_groups(self):
        with pytest.raises(ValueError):
            fpr_ratio([0, 0, 1], [0.1, 0.2, 0.9], [20, 30, 60], 0.5)


class TestPrecisionRatio:
    def test_the_age_group_with_the_higher_precision_is_the_denominator(self):
        # 50 or over: 3 flagged, 2 of them Fraudulent (2/3). Under 50: 3 flagged, 1 (1/3).
        # Unflagged Applications never count, whatever their label.
        labels = [1, 1, 0, 1, 1, 0, 0, 0, 1]
        scores = [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.7, 0.1, 0.2]
        ages = [60, 70, 80, 90, 20, 30, 40, 40, 30]

        assert precision_ratio(labels, scores, ages, 0.5) == pytest.approx(0.5)

    def test_age_50_belongs_to_the_older_group(self):
        # Flagged: a Fraudulent Applicant aged 49 or 50, one Legitimate under 50, one
        # Fraudulent and one Legitimate aged 60. At 49, under 50 is 1/2 and 50 or over 1/2;
        # at 50, under 50 is 0/1 and 50 or over 2/3.
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.9, 0.9, 0.9]

        assert precision_ratio(labels, scores, [49, 30, 60, 60], 0.5) == 1.0
        assert precision_ratio(labels, scores, [50, 30, 60, 60], 0.5) == 0.0

    def test_no_fraudulent_application_flagged_in_either_group_counts_as_equal(self):
        assert precision_ratio([0, 0, 1], [0.9, 0.9, 0.1], [20, 60, 30], 0.5) == 1.0

    def test_is_undefined_when_an_age_group_has_nothing_flagged(self):
        assert np.isnan(precision_ratio([1, 0], [0.9, 0.1], [60, 20], 0.5))

    def test_needs_applications_in_both_age_groups(self):
        with pytest.raises(ValueError, match="no Applications under 50"):
            precision_ratio([1, 0], [0.9, 0.9], [60, 70], 0.5)

    def test_rejects_missing_ages(self):
        with pytest.raises(ValueError):
            precision_ratio([1, 0], [0.9, 0.9], [float("nan"), 60], 0.5)


class TestPairedBootstrap:
    """Seam 3: the paired bootstrap over the Test Months."""

    # 160 Legitimate Applications scored below every one of 40 Fraudulent ones, half of
    # each Age Group: a perfect ranking, whatever the resample.
    LABELS = [0] * 160 + [1] * 40
    AGES = [30, 60] * 100
    PERFECT = [i / 1000 for i in range(160)] + [0.9 + i / 1000 for i in range(40)]

    @staticmethod
    def _noisy(n=600, seed=7):
        rng = np.random.default_rng(seed)
        labels = (rng.random(n) < 0.15).astype(int)
        ages = rng.choice([20, 30, 40, 50, 60, 70], size=n)
        good = labels + rng.normal(0, 0.8, n)
        weak = labels + rng.normal(0, 2.0, n)
        return labels, ages, {"good": good, "weak": weak}

    @staticmethod
    def _bootstrap(labels, ages, scores, seed=0, thresholds=None):
        thresholds = thresholds or {n: threshold_for_fpr(labels, s) for n, s in scores.items()}
        return paired_bootstrap(
            labels, ages, scores, thresholds, reference="good", n_resamples=200, seed=seed
        )

    def test_a_perfect_ranking_has_intervals_of_exactly_one(self):
        # Every resample ranks every Fraudulent Application first, and a Threshold between
        # the two classes flags no Legitimate one: each metric is 1 on every resample.
        result = self._bootstrap(
            self.LABELS, self.AGES, {"good": self.PERFECT}, thresholds={"good": 0.5}
        )

        intervals = result["good"]["interval"]
        assert set(intervals) == {"recall_at_5pct_fpr", "pr_auc", "fpr_ratio", "precision_ratio"}
        for interval in intervals.values():
            assert interval == pytest.approx([1.0, 1.0], abs=1e-12)
        assert "difference" not in result["good"]

    def test_identical_scores_differ_by_exactly_zero(self):
        labels, ages, scores = self._noisy()
        copies = {"good": scores["good"], "copy": scores["good"].copy()}

        result = self._bootstrap(labels, ages, copies)

        assert all(interval == [0.0, 0.0] for interval in result["copy"]["difference"].values())

    def test_the_same_seed_gives_the_same_intervals(self):
        labels, ages, scores = self._noisy()

        assert self._bootstrap(labels, ages, scores, 3) == self._bootstrap(labels, ages, scores, 3)
        assert self._bootstrap(labels, ages, scores, 3) != self._bootstrap(labels, ages, scores, 4)

    def test_a_clearly_better_model_has_a_difference_interval_above_zero(self):
        labels, ages, scores = self._noisy()

        low, _ = self._bootstrap(labels, ages, scores)["weak"]["difference"]["recall_at_5pct_fpr"]

        assert low > 0
