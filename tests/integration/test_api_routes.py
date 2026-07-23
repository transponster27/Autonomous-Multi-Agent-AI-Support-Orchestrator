import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.testclient")

"""Integration tests for FastAPI routes."""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes import router

# Create a test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_status_endpoint():
    """Test that the status endpoint returns valid JSON."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_count" in data

def test_query_endpoint_fallback():
    """Test that /query handles requests gracefully even with no docs."""
    response = client.get("/query?q=hello+there&session_id=test")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "agent_type" in data
    assert "sources" in data

def test_query_endpoint_validation_fields():
    """Test that validation metrics are returned in the response."""
    response = client.get("/query?q=what+is+this&session_id=test")
    data = response.json()
    
    # Even if it's conversational, these fields should exist
    assert "validation_score" in data
    assert "validation_attempts" in data