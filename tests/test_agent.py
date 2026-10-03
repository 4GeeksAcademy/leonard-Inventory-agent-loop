import copy
import csv
import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessage

from agent import ConversationLog, InventoryAgent, LOG_FIELDS, TOOLS, llm_settings
from api.app import create_app


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=ChatCompletionMessage.model_validate(response))])


def tool_message(*calls):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": arguments}}
            for index, (name, arguments) in enumerate(calls)
        ],
    }


def test_multistep_loop_and_append_only_sessions(tmp_path):
    log_path = tmp_path / "conversation_log.csv"
    llm = FakeLLM([
        tool_message(("add_product", '{"name":"Oat milk","quantity":5,"unit":"liters"}')),
        tool_message(("get_low_stock_alerts", "{}")),
        {"role": "assistant", "content": "Added oat milk. Its 5 liters are below the threshold."},
        {"role": "assistant", "content": "You added oat milk earlier."},
    ])
    with TestClient(create_app(tmp_path / "products.csv")) as api:
        agent = InventoryAgent(llm, api, ConversationLog(log_path), "test-model")
        assert "5 liters" in agent.respond("Add 5 liters of oat milk and check alerts")
        first_result = llm.requests[1]["messages"][-1]
        assert first_result["role"] == "tool"
        assert first_result["tool_call_id"] == "call_0"
        assert json.loads(first_result["content"])["data"]["id"] == 1
        assert json.loads(llm.requests[2]["messages"][-1]["content"])["data"][0]["quantity"] == 5
        agent.respond("What did I add earlier?")
        assert len([message for message in llm.requests[3]["messages"] if message["role"] == "user"]) == 2
        previous_bytes = log_path.read_bytes()
        new_agent = InventoryAgent(FakeLLM([{"role": "assistant", "content": "Hello Carla."}]), api, ConversationLog(log_path), "test-model")
        new_agent.respond("Hello")
        assert log_path.read_bytes().startswith(previous_bytes)
        assert len(new_agent.messages) == 3
    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == LOG_FIELDS
        rows = list(reader)
    assert [row["actor"] for row in rows[:6]] == ["user", "agent", "tool", "agent", "tool", "agent"]
    assert rows[1]["tool_call"] == rows[2]["tool_call"] == "add_product"
    assert rows[5]["tool_call"] == ""
    assert all(datetime.fromisoformat(row["timestamp"]).tzinfo for row in rows)


def test_all_tools_and_error_results(tmp_path):
    with TestClient(create_app(tmp_path / "products.csv")) as api:
        agent = InventoryAgent(None, api, ConversationLog(tmp_path / "log.csv"), "test")
        assert agent.execute_tool("add_product", '{"name":"Arabica","quantity":20,"unit":"bags","location":"Downtown"}')["status_code"] == 201
        assert len(agent.execute_tool("list_inventory", '{"location":"Downtown"}')["data"]) == 1
        assert agent.execute_tool("update_stock", '{"product_id":1,"delta":-12}')["data"]["quantity"] == 8
        assert agent.execute_tool("get_low_stock_alerts", "{}")["data"][0]["quantity"] == 8
        assert agent.execute_tool("update_stock", '{"product_id":1,"delta":-9}')["status_code"] == 409
        assert agent.execute_tool("update_stock", '{"product_id":999,"delta":1}')["status_code"] == 404
        assert "error" in agent.execute_tool("unknown", "{}")
        for arguments in ("not json", "[]", '{"product_id":1,"delta":"NaN"}', '{"product_id":true,"delta":1}'):
            assert "error" in agent.execute_tool("update_stock", arguments)
    assert {tool["function"]["name"] for tool in TOOLS} == {"add_product", "list_inventory", "update_stock", "get_low_stock_alerts"}
    assert all(tool["function"]["parameters"]["type"] == "object" for tool in TOOLS)


def test_multiple_tool_calls_errors_and_loop_limit(tmp_path):
    llm = FakeLLM([tool_message(("unknown", "{}"), ("list_inventory", "invalid json"))])
    agent = InventoryAgent(llm, None, ConversationLog(tmp_path / "log.csv"), "test", max_rounds=1)
    assert "tool-call limit" in agent.respond("Check inventory")
    assert [message["role"] for message in agent.messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert all("error" in json.loads(message["content"]) for message in agent.messages[3:5])


def test_network_and_model_failures(tmp_path):
    def unavailable(request):
        raise httpx.ConnectError("offline", request=request)
    with httpx.Client(transport=httpx.MockTransport(unavailable), base_url="http://inventory") as api:
        agent = InventoryAgent(FakeLLM([OpenAIError("offline")]), api, ConversationLog(tmp_path / "log.csv"), "test")
        result = agent.execute_tool("update_stock", '{"product_id":1,"delta":1}')
        assert result["outcome_uncertain"] is True
        assert "unavailable" in agent.respond("Hello")
        assert agent.messages[-1]["role"] == "assistant"


def test_csv_quoting_and_existing_header(tmp_path):
    path = tmp_path / "log.csv"
    text = 'Carla said, "hello"\nNew line'
    ConversationLog(path).append("user", text)
    ConversationLog(path).append("agent", "Hello")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["message"] == text
    assert len(rows) == 2


def test_provider_settings(monkeypatch):
    for name in ("NVIDIA_API_KEY", "NVIDIA_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        llm_settings()
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    assert llm_settings() == ("fake-groq-key", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia-key")
    assert llm_settings() == ("fake-nvidia-key", "https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3.5-lightning-30b-a3b")
    monkeypatch.setenv("NVIDIA_MODEL", "another-tool-capable-model")
    assert llm_settings()[2] == "another-tool-capable-model"
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    assert llm_settings() == ("fake-nvidia-key", "https://example.test/v1", "custom-model")