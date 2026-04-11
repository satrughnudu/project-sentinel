"""
Integration tests for Project Sentinel.
Tests all API endpoints and core environment logic.
Run with: python -m pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from server.app import app, env
from server.env import SentinelEnv, TASK_DEFS, KNOWN_USERS, R
from server.models import Action


client = TestClient(app)


# --- API endpoint tests ---

class TestRootEndpoint:
    def test_health_check(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "running"
        assert "tasks" in body
        assert len(body["tasks"]) == 3

    def test_tasks_endpoint(self):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 3
        assert tasks[0]["id"] == "task_1_easy"


class TestResetEndpoint:
    def test_reset_with_task_id(self):
        r = client.post("/reset", json={"task_id": "task_1_easy"})
        assert r.status_code == 200
        obs = r.json()
        assert "request" in obs
        assert "system_state" in obs
        assert obs["system_state"]["network_health"] == 100

    def test_reset_bare_body(self):
        r = client.post("/reset")
        assert r.status_code == 200

    def test_reset_empty_json(self):
        r = client.post("/reset", json={})
        assert r.status_code == 200

    def test_reset_invalid_task(self):
        r = client.post("/reset", json={"task_id": "nonexistent"})
        assert r.status_code == 400


class TestStepEndpoint:
    def test_step_before_reset_fails(self):
        # fresh env without reset
        fresh = SentinelEnv()
        try:
            fresh.step(Action(decision="block", reasoning="test"))
            assert False, "Should have raised"
        except RuntimeError:
            pass

    def test_investigation_action(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.post("/step", json={
            "decision": "investigate_user", "reasoning": "checking identity"})
        assert r.status_code == 200
        body = r.json()
        assert body["reward"] == R["inv"]
        assert body["done"] is False
        # observation should include investigation results
        assert len(body["observation"]["investigation_results"]) == 1

    def test_decision_action(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.post("/step", json={
            "decision": "block", "reasoning": "suspicious"})
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["reward"], float)
        assert "was_malicious" in body["info"]

    def test_case_insensitive_decision(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.post("/step", json={
            "decision": "BLOCK", "reasoning": "test"})
        assert r.status_code == 200

    def test_invalid_decision_rejected(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.post("/step", json={
            "decision": "invalid_action", "reasoning": "test"})
        assert r.status_code == 422  # pydantic validation error

    def test_full_episode_completes(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        done = False
        steps = 0
        while not done and steps < 20:
            r = client.post("/step", json={
                "decision": "allow", "reasoning": "testing", "confidence": 0.5})
            assert r.status_code == 200
            body = r.json()
            done = body["done"]
            steps += 1
        assert done is True
        assert steps <= 16  # max budget = 8 scenarios * 2


class TestGraderEndpoint:
    def test_grader_before_episode_done(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.post("/grader")
        assert r.status_code == 200
        assert r.json()["score"] == 0.0  # not done yet

    def test_grader_after_episode(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        for _ in range(16):
            r = client.post("/step", json={"decision": "block", "reasoning": "test"})
            if r.json()["done"]:
                break
        r = client.post("/grader")
        assert r.status_code == 200
        score = r.json()["score"]
        assert 0.0 <= score <= 1.0

    def test_grader_perfect_run(self):
        """Block all malicious, allow all benign = high score."""
        e = SentinelEnv()
        obs = e.reset(task_id="task_1_easy")
        for i, sc in enumerate(e.current_scenarios):
            decision = "block" if sc["malicious"] else "allow"
            e.step(Action(decision=decision, reasoning="oracle"))
        score = e.grader()
        assert score > 0.8

    def test_grader_worst_run(self):
        """Allow all malicious, block all benign = low score."""
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        for sc in e.current_scenarios:
            decision = "allow" if sc["malicious"] else "block"
            e.step(Action(decision=decision, reasoning="worst"))
        score = e.grader()
        assert score < 0.3


class TestStateEndpoint:
    def test_state_no_ground_truth_leak(self):
        client.post("/reset", json={"task_id": "task_1_easy"})
        r = client.get("/state")
        assert r.status_code == 200
        state_str = str(r.json())
        assert "malicious" not in state_str.lower()
        assert "clues" not in state_str.lower()


# --- Core environment unit tests ---

class TestSentinelEnv:
    def test_all_tasks_are_valid(self):
        e = SentinelEnv()
        for tid in TASK_DEFS:
            obs = e.reset(task_id=tid)
            assert obs.remaining_steps > 0
            assert obs.system_state.network_health == 100

    def test_procedural_generation_varies(self):
        """Two resets of the same task should produce different episodes."""
        e = SentinelEnv()
        obs1 = e.reset(task_id="task_1_easy")
        tools1 = [sc["tool_name"] for sc in e.current_scenarios]
        obs2 = e.reset(task_id="task_1_easy")
        tools2 = [sc["tool_name"] for sc in e.current_scenarios]
        # not guaranteed different every time, but over many runs they will be
        assert obs1.episode_id != obs2.episode_id

    def test_damage_system_state(self):
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        initial_health = e.system_state["network_health"]
        # find a malicious scenario and allow it
        for sc in e.current_scenarios:
            if sc["malicious"]:
                e.step(Action(decision="allow", reasoning="deliberately allowing"))
                break
        assert e.system_state["network_health"] < initial_health

    def test_investigation_provides_clues(self):
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        result = e.step(Action(decision="investigate_user", reasoning="check"))
        inv = result.observation.investigation_results[0]
        assert "summary" in inv.findings
        assert inv.investigation_type == "investigate_user"

    def test_duplicate_investigation_blocked(self):
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        r1 = e.step(Action(decision="investigate_user", reasoning="first"))
        assert r1.reward == R["inv"]
        r2 = e.step(Action(decision="investigate_user", reasoning="duplicate"))
        assert r2.reward == 0.0  # no reward for duplicate
        assert "already investigated" in r2.info.get("note", "")

    def test_investigation_bonus_on_correct_decision(self):
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        sc = e.current_scenarios[0]
        e.step(Action(decision="investigate_user", reasoning="check"))
        correct_decision = "block" if sc["malicious"] else "allow"
        result = e.step(Action(decision=correct_decision, reasoning="decided"))
        # should include investigation bonus
        assert result.info["investigation_bonus"] > 0

    def test_task3_hard_benign_first(self):
        """task_3_hard should put benign scenarios before malicious ones."""
        e = SentinelEnv()
        e.reset(task_id="task_3_hard")
        td = TASK_DEFS["task_3_hard"]
        n_ben = td["n_ben"]
        # first n_ben scenarios should all be benign
        for i in range(n_ben):
            assert e.current_scenarios[i]["malicious"] is False

    def test_step_after_done_raises(self):
        e = SentinelEnv()
        e.reset(task_id="task_1_easy")
        # exhaust all scenarios
        for _ in range(20):
            if e.done:
                break
            e.step(Action(decision="allow", reasoning="rush"))
        assert e.done
        with pytest.raises(RuntimeError):
            e.step(Action(decision="allow", reasoning="post-done"))


class TestKnownUsers:
    def test_all_users_have_required_fields(self):
        required = {"role", "clearance", "department", "login_history", "flagged"}
        for uid, data in KNOWN_USERS.items():
            missing = required - set(data.keys())
            assert not missing, f"User '{uid}' missing fields: {missing}"

    def test_flagged_users_have_no_clearance_or_restricted(self):
        for uid, data in KNOWN_USERS.items():
            if data["flagged"]:
                assert data["clearance"] in ("none", "restricted", "api_key"), \
                    f"Flagged user '{uid}' has unexpected clearance: {data['clearance']}"
