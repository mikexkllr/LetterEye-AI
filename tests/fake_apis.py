"""A tiny local server that speaks the OpenAI and Anthropic wire formats, for testing the real LangChain
adapters end to end without network access or keys.

Its "model" behaves like a sensible decision model: it prefers options whose text appears in the STATE.
"""

from __future__ import annotations

import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FACTS = {"sender_organization": "Stadtwerke München", "sender_person": "", "recipient_name": "Bob Smith",
         "letter_date": "14.03.2026", "subject": "Jahresabrechnung Strom"}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _options(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    state = prompt.split("<<<", 1)[1].split(">>>", 1)[0] if "<<<" in prompt else ""
    block = prompt.split("OPTIONS:", 1)[1] if "OPTIONS:" in prompt else ""
    return state, re.findall(r"^([A-T])\) (.*)$", block, flags=re.MULTILINE)


def _scores(prompt: str) -> dict[str, float]:
    state, options = _options(prompt)
    weights = {}
    question = re.search(r"QUESTION: Is this letter addressed to (.*)\?", prompt)
    if question:  # the yes/no double check
        found = any(name.strip() in state for name in question.group(1).split(" / "))
        return {label: (0.95 if (option == "Yes") == found else 0.05) for label, option in options}
    for label, option in options:
        name = option.split(" (also")[0].split(":")[0]
        weights[label] = 20.0 if name and name in state else 1.0
    total = sum(weights.values()) or 1.0
    return {label: w / total for label, w in weights.items()}


class FakeApi:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.logprobs = True  # set False to act like a server that ignores `logprobs`
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeApi:
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ------------------------------------------------------------------ responses
    def openai(self, body: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        prompt = _text_of(body["messages"][-1]["content"])
        message: dict[str, Any] = {"role": "assistant", "content": None}
        choice: dict[str, Any] = {"index": 0, "finish_reason": "stop"}
        if body.get("tools"):
            function = body["tools"][0]["function"]
            arguments = FACTS if function["name"] == "LetterFacts" else self._structured(prompt)
            message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                "name": function["name"], "arguments": json.dumps(arguments)}}]
            choice["finish_reason"] = "tool_calls"
        elif "OPTIONS:" in prompt:
            scores = _scores(prompt)
            best = max(scores, key=scores.get)
            message["content"] = best
            if body.get("logprobs") and self.logprobs:
                top = [{"token": label, "logprob": math.log(p), "bytes": [ord(label)]}
                       for label, p in sorted(scores.items(), key=lambda kv: -kv[1])][: body.get("top_logprobs", 5)]
                choice["logprobs"] = {"content": [{"token": best, "logprob": math.log(scores[best]),
                                                   "bytes": [ord(best)], "top_logprobs": top}]}
        else:
            message["content"] = "Stadtwerke München GmbH\nHerrn\nBob Smith\nSehr geehrter Herr Smith,\n"
        choice["message"] = message
        result = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
                  "choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        if body.get("stream"):
            text = message["content"] or ""
            chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                       "choices": [{"index": 0, "delta": {"content": line + "\n"}, "finish_reason": None}]}
                      for line in text.splitlines()]
            chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            return chunks
        return result

    def anthropic(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt = _text_of(body["messages"][-1]["content"])
        if body.get("tools"):
            tool = body["tools"][0]
            arguments = FACTS if tool["name"] == "LetterFacts" else self._structured(prompt)
            content = [{"type": "tool_use", "id": "toolu_1", "name": tool["name"], "input": arguments}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "B"}]
            stop = "end_turn"
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": content,
                "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}

    def anthropic_stream(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        lines = ["Stadtwerke München GmbH", "Herrn", "Bob Smith", "Sehr geehrter Herr Smith,"]
        message = {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [],
                   "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}
        return [
            {"type": "message_start", "message": message},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            *({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": line + "\n"}}
              for line in lines),
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 20}},
            {"type": "message_stop"},
        ]

    @staticmethod
    def _structured(prompt: str) -> dict[str, Any]:
        scores = _scores(prompt)
        return {"answer": max(scores, key=scores.get), **{label: round(p, 4) for label, p in scores.items()}}

    def _handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, payload: Any, status: int = 200) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                api.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                if self.path.startswith("/v1/models"):
                    self._send({"data": [{"id": "fake-decider"}, {"id": "fake-vision"}], "object": "list"})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                api.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                     "body": body})
                if self.path.endswith("/chat/completions"):
                    result = api.openai(body)
                    if isinstance(result, list):  # server-sent events
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for chunk in result:
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    self._send(result)
                elif self.path.endswith("/v1/messages") and body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for event in api.anthropic_stream(body):
                        self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                elif self.path.endswith("/v1/messages"):
                    self._send(api.anthropic(body))
                else:
                    self._send({"error": "not found"}, 404)

        return Handler
