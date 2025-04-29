import pytest
from fastapi.testclient import TestClient
from backend import app  # make sure backend.py is in the same directory

client = TestClient(app)

def test_valid_company_name():
    response = client.post("/predict/", json={"company_name": "NATCOPHARM"})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_company_name():
    response = client.post("/predict/", json={"company_name": "ασδφασδφ"})
    assert response.status_code == 404  # Company not found is expected
    assert "detail" in response.json()


def test_empty_input():
    response = client.post("/predict/", json={})
    assert response.status_code == 422  # Missing required field
