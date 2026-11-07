"""The real LangChain adapters (langchain-openai, langchain-anthropic) over HTTP against a local fake API."""

import pytest
from PIL import Image

from lettereye.ai import factory
from lettereye.pipeline.processor import Directory, Processor
from lettereye.settings import ProviderConfig, Settings

from .conftest import CLEAN_LETTER, FakeOCR
from .fake_apis import FakeApi


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with FakeApi() as server:
        yield server


def settings_for(provider: ProviderConfig, **extra) -> Settings:
    return Settings(providers=[provider], decision_provider=provider.id, decision_model="fake-decider",
                    ocr_llm_provider=provider.id, ocr_llm_model="fake-vision", language="de", **extra)


def openai_provider(api) -> ProviderConfig:
    return ProviderConfig(id="local-api", name="Local API", kind="openai", base_url=f"{api.url}/v1")


def test_openai_compatible_decisions_use_token_probabilities(api):
    engine = factory.build_decision_engine(settings_for(openai_provider(api)), lambda _id: "")
    result = engine.choice("Herrn Bob Smith, Sehr geehrter Herr Smith", "Who is the addressee?",
                           ["Alice Johnson", "Bob Smith", "Someone else (not in this list)"])
    assert result.choice == "Bob Smith" and result.confidence > 0.8
    assert result.method == "logprobs"
    body = api.requests[-1]["body"]
    assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["temperature"] == 0
    assert body["max_completion_tokens"] == 12


def test_openai_compatible_server_without_logprobs_falls_back_to_structured(api):
    api.logprobs = False
    engine = factory.build_decision_engine(settings_for(openai_provider(api)), lambda _id: "")
    result = engine.choice("Dear Bob Smith", "Who?", ["Alice Johnson", "Bob Smith"])
    assert result.choice == "Bob Smith" and result.method == "structured"
    assert api.requests[-1]["body"]["tools"][0]["function"]["name"] == "Decision2"


def test_anthropic_decisions_and_extraction_use_tool_calls(api):
    provider = ProviderConfig(id="claude", name="Claude", kind="anthropic", base_url=api.url)
    settings = settings_for(provider)
    keys = {"claude": "sk-ant-test"}.get
    result = factory.build_decision_engine(settings, keys).choice("Dear Bob Smith", "Who?", ["Bob Smith", "Carol"])
    assert result.choice == "Bob Smith" and result.method == "structured"
    request = api.requests[-1]
    assert request["path"] == "/v1/messages"
    headers = request["headers"]  # the SDK sends the key as x-api-key or as a bearer token, depending on version
    assert headers.get("x-api-key") == "sk-ant-test" or headers.get("authorization") == "Bearer sk-ant-test"
    facts = factory.build_extractor(settings, keys).extract(CLEAN_LETTER)
    assert facts.sender == "Stadtwerke München" and facts.subject == "Jahresabrechnung Strom"


def test_ocr_llm_sends_the_page_image_and_streams_the_text(api):
    ocr = factory.build_llm_ocr(settings_for(openai_provider(api)), lambda _id: "")
    result = ocr.recognize([Image.new("RGB", (200, 280), "white")])
    assert "Bob Smith" in result.text and result.device == "Local API"
    content = api.requests[-1]["body"]["messages"][-1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_ocr_llm_on_anthropic_streams_content_blocks(api):
    provider = ProviderConfig(id="claude", name="Claude", kind="anthropic", base_url=api.url)
    ocr = factory.build_llm_ocr(settings_for(provider), {"claude": "sk-ant-test"}.get)
    result = ocr.recognize([Image.new("RGB", (200, 280), "white")])
    assert result.text.splitlines() == ["Stadtwerke München GmbH", "Herrn", "Bob Smith", "Sehr geehrter Herr Smith,"]
    image = api.requests[-1]["body"]["messages"][-1]["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"


def test_whole_pipeline_on_an_openai_compatible_api(api, db, scan_image):
    worker = db.create_worker("John Doe")
    db.create_recipient(worker.id, "Bob Smith")
    db.create_recipient(worker.id, "Eve Adams")
    settings = settings_for(openai_provider(api), ocr_mode="fast_only")
    processor = Processor(settings, factory.build_decision_engine(settings, lambda _id: ""),
                          factory.build_extractor(settings, lambda _id: ""), FakeOCR(CLEAN_LETTER), None,
                          lambda: Directory(db.list_workers(), db.list_recipients()))
    analysis = processor.analyze(scan_image)
    assert analysis.status == "filed"
    assert analysis.decision.recipient.name == "Bob Smith" and analysis.decision.worker.name == "John Doe"
    assert analysis.facts.sender == "Stadtwerke München"
    assert {d["method"] for d in analysis.decision.decisions} == {"logprobs"}
