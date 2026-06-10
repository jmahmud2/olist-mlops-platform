import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.serve import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_forecast_endpoint():
    response = client.post("/forecast/daily", json={"days": 7})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 7

def test_support_shipping_intent():
    response = client.post("/support", json={"message": "Where is my order?"})
    assert response.status_code == 200
    assert response.json()["intent"] == "shipping"

def test_support_complaint_escalation():
    response = client.post("/support", json={"message": "This is terrible! I want a manager"})
    assert response.status_code == 200
    assert response.json()["escalate"] == True