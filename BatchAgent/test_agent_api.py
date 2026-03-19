import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import the router to test
from BatchAgent.agent_route_fastapi import router, _tasks

# Create a minimal FastAPI app mapped to the router for testing
app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_agent_health():
    response = client.get("/agent/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "BatchAgent"}


def test_get_active_tools():
    response = client.get("/agent/tools?strategy=hybrid")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)


@patch("BatchAgent.agent_route_fastapi._run_blocking")
def test_agent_run_endpoint(mock_run_blocking):
    # Mock the internal run execution to just return true and not call the LLM
    async def mock_run(*args, **kwargs):
        record = args[0]
        record.success = True
        record.status = "done"
        return True
    mock_run_blocking.side_effect = mock_run

    payload = {
        "goal": "Test goal",
        "tool_strategy": "hybrid",
        "provider": "openai",
        "model": "qwen3.5-9b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY"
    }

    # Simulate anonymous request (no auth header)
    response = client.post("/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "task_id" in data
    assert data["status"] == "done"
    assert data["success"] is True
    
    task_id = data["task_id"]
    
    # Test getting status for the created task
    status_response = client.get(f"/agent/status/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] == "done"


def test_agent_init_session():
    payload = {
        "goal": "Empty initialized task"
    }
    response = client.post("/agent/sessions/init", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "task_id" in data
    assert "workspace_dir" in data
    assert data["goal"] == "Empty initialized task"
    assert data["status"] == "pending"


def test_agent_status_not_found():
    response = client.get("/agent/status/nonexistent_task_id")
    assert response.status_code == 404


def test_list_sessions():
    response = client.get("/agent/sessions?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)

@patch("BatchAgent.agent_route_fastapi._run_streaming")
def test_agent_stream_endpoint(mock_run_stream):
    async def mock_stream(*args, **kwargs):
        record = args[0]
        record.success = True
        record.status = "done"
        return True
    mock_run_stream.side_effect = mock_stream
    payload = {
        "goal": "Test stream",
        "tool_strategy": "hybrid"
    }
    response = client.post("/agent/stream", json=payload)
    assert response.status_code == 200

def test_session_turns_not_found():
    response = client.get("/agent/sessions/nonexistent/turns")
    assert response.status_code == 404

def test_delete_session_not_found():
    response = client.delete("/agent/sessions/nonexistent")
    assert response.status_code == 404

def test_files_content_not_found():
    response = client.get("/agent/sessions/nonexistent/files/content?name=file.md")
    assert response.status_code == 404

def test_files_content_put_not_found():
    response = client.put("/agent/sessions/nonexistent/files/content", json={"name": "file.md", "content": "text"})
    assert response.status_code == 404

def test_resources_attach_not_found():
    response = client.post("/agent/sessions/nonexistent/resources/attach", json={"url": "http://example.com"})
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main(["-v", __file__])
