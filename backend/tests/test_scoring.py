"""Unit tests for scoring.py — candidate scoring and feedback builders."""
import pytest
from app.scoring import candidate_score, update_best, build_leak_feedback, build_utility_feedback


class TestCandidateScore:
    def test_no_leak_full_utility(self):
        config = {"weights": {"privacy": 0.6, "utility": 0.4}}
        score = candidate_score(config, leaked_count=0, total=5, task_utility=1.0)
        assert score == pytest.approx(1.0)  # 0.6*1.0 + 0.4*1.0

    def test_all_leaked_no_utility(self):
        config = {"weights": {"privacy": 0.6, "utility": 0.4}}
        score = candidate_score(config, leaked_count=5, total=5, task_utility=0.0)
        assert score == pytest.approx(0.0)

    def test_partial_leak(self):
        config = {"weights": {"privacy": 0.6, "utility": 0.4}}
        score = candidate_score(config, leaked_count=2, total=5, task_utility=0.5)
        # privacy = 1 - 2/5 = 0.6, weighted = 0.6 * 0.6 = 0.36
        # utility = 0.5, weighted = 0.4 * 0.5 = 0.2
        # total = 0.56
        assert score == pytest.approx(0.56)

    def test_zero_total_no_crash(self):
        config = {"weights": {"privacy": 0.6, "utility": 0.4}}
        score = candidate_score(config, leaked_count=0, total=0, task_utility=0.5)
        # privacy = 1.0 (no division by zero), weighted = 0.6
        # utility = 0.5, weighted = 0.2
        assert score == pytest.approx(0.8)

    def test_none_utility_treated_as_zero(self):
        config = {"weights": {"privacy": 0.6, "utility": 0.4}}
        score = candidate_score(config, leaked_count=0, total=5, task_utility=None)
        assert score == pytest.approx(0.6)  # 0.6*1.0 + 0.4*0.0


class TestUpdateBest:
    def test_none_prev_returns_cand(self):
        cand = {"text": "test", "score": 0.8}
        result = update_best(None, cand)
        assert result == cand

    def test_better_score_updates(self):
        prev = {"text": "old", "score": 0.5}
        cand = {"text": "new", "score": 0.8}
        result = update_best(prev, cand)
        assert result == cand

    def test_worse_score_keeps_prev(self):
        prev = {"text": "old", "score": 0.8}
        cand = {"text": "new", "score": 0.5}
        result = update_best(prev, cand)
        assert result == prev

    def test_equal_score_keeps_prev(self):
        prev = {"text": "old", "score": 0.5}
        cand = {"text": "new", "score": 0.5}
        result = update_best(prev, cand)
        assert result == prev


class TestBuildLeakFeedback:
    def test_single_leak(self):
        details = [{
            "attribute": "name",
            "guess": "John Smith",
            "evidence": ["John S."],
            "rationale": "Name is directly stated",
        }]
        feedback = build_leak_feedback(details)
        assert "STILL inferable" in feedback
        assert "name" in feedback
        assert "John Smith" in feedback
        assert "John S." in feedback

    def test_multiple_leaks(self):
        details = [
            {"attribute": "name", "guess": "Alice", "evidence": [], "rationale": ""},
            {"attribute": "age", "guess": "30", "evidence": ["thirty"], "rationale": ""},
        ]
        feedback = build_leak_feedback(details)
        assert "name" in feedback
        assert "age" in feedback
        assert "Alice" in feedback
        assert "30" in feedback

    def test_empty_evidence(self):
        details = [{"attribute": "name", "guess": "John", "evidence": [], "rationale": ""}]
        feedback = build_leak_feedback(details)
        assert "(no specific span)" in feedback


class TestBuildUtilityFeedback:
    def test_basic_feedback(self):
        j = {
            "task_utility": 0.3,
            "factual_consistency": 0.8,
            "format_preserved": 0.9,
            "notes": "Too generic",
        }
        feedback = build_utility_feedback(j)
        assert "No attribute leaked" in feedback
        assert "task_utility=0.30" in feedback
        assert "Too generic" in feedback
        assert "LIGHTLY" in feedback

    def test_with_privacy_note(self):
        j = {
            "task_utility": 0.4,
            "factual_consistency": 0.7,
            "format_preserved": 1.0,
            "notes": "Lost details",
        }
        feedback = build_utility_feedback(j, privacy_note="all attributes hidden")
        assert "all attributes hidden" in feedback
        assert "keep it that way" in feedback
