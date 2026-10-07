import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
HEALTHY = {"air_temp_k": 300, "process_temp_k": 310.5, "rotational_speed_rpm": 1550,
           "torque_nm": 38, "tool_wear_min": 20}
RISKY = {**HEALTHY, "torque_nm": 62, "tool_wear_min": 230}     # worn tool under high torque


# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_threshold():                            # checks TODO 3c
    body = client.get("/model-info").json()
    assert {"model_source", "threshold", "features"} <= set(body)


def test_predict_returns_valid_response():
    r = client.post("/predict", json=HEALTHY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["failure_probability"] <= 1
    assert isinstance(body["failure_predicted"], bool)


def test_risky_machine_scores_higher_than_healthy():
    healthy = client.post("/predict", json=HEALTHY).json()
    risky = client.post("/predict", json=RISKY).json()
    assert risky["failure_probability"] > healthy["failure_probability"]
    assert risky["failure_predicted"] is True


def test_actions_are_filled_in_and_differ():                    # checks TODO 3b
    healthy = client.post("/predict", json=HEALTHY).json()["recommended_action"]
    risky = client.post("/predict", json=RISKY).json()["recommended_action"]
    assert "TODO" not in (healthy + risky)
    assert healthy != risky


# ---------- your tests ----------

def test_out_of_range_input_is_rejected():
    # Test below minimum bound (negative torque)
    r_low = client.post("/predict", json={**HEALTHY, "torque_nm": -5})
    assert r_low.status_code == 422, f"Expected 422 for low value, got {r_low.status_code}"

    # Test above maximum bound (e.g., extremely high torque)
    r_high = client.post("/predict", json={**HEALTHY, "torque_nm": 10000})
    assert r_high.status_code == 422, f"Expected 422 for high value, got {r_high.status_code}"

#def test_out_of_range_input_is_rejected_template():
    ## TODO 4a: send a reading with an impossible value and assert the API answers 422.
    ## HINT: r = client.post("/predict", json={**HEALTHY, "torque_nm": -5}); assert r.status_code == 422
    ## THINK: test both edges? (a value just below your minimum AND just above your maximum)
    #assert False, "TODO 4a: write this test"

def test_missing_field_is_rejected():
    # Make a copy and remove a required field
    reading = dict(HEALTHY)
    reading.pop("tool_wear_min")

    # Send request and assert that the API returns 422
    response = client.post("/predict", json=reading)
    assert response.status_code == 422, f"Expected status code 422, but got {response.status_code}"

#def test_missing_field_is_rejected_template():
    ## TODO 4b: remove one field from HEALTHY, send it, and assert 422.
    ## HINT: reading = dict(HEALTHY); reading.pop("tool_wear_min")
    #reading = dict(HEALTHY); reading.pop("tool_wear_min")
    #assert False, "TODO 4b: write this test"


def test_my_own_idea():
    # TODO 4c (optional, recommended): write a test of your own. Ideas:
    #   - more tool wear never LOWERS the failure probability (a "behaviour" test)
    #   - a text value such as "torque_nm": "high" is rejected with 422
    #   - /predict answers in under 200 ms
    #   - the API never predicts failure for a brand-new tool at normal torque
    pytest.skip("TODO 4c: optional")
