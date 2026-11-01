"""
Minimal local mock of an OpenAI-compatible /v1/chat/completions endpoint,
used ONLY to validate llm_client.py's real HTTP request/response handling
without needing Ollama itself installed (not possible in the sandbox this
was built in -- ollama.com isn't reachable from there). This is not part
of the shipped pipeline; it's a one-off validation tool.

Run: python3 -m llm_stage.mock_openai_server (starts on port 11500, Ctrl+C to stop)
"""

import http.server
import json
import threading


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        wants_tool_call = self.server.respond_with_tool_call  # type: ignore[attr-defined]

        if wants_tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": json.dumps({
                            "is_vulnerable": True, "confidence": 0.81,
                            "cwe": "CWE-78", "severity": "high",
                            "taint_source": "function parameter, no visible caller",
                            "explanation": "mock server response",
                            "suggested_fix": "validate input",
                        }),
                    },
                }],
            }
        else:
            # Simulate a smaller local model that ignores tool_choice and
            # just answers in plain text with embedded JSON -- exercises
            # the fallback path in _call_openai_compatible.
            message = {
                "role": "assistant",
                "content": ('Here is my assessment:\n'
                            '{"is_vulnerable": false, "confidence": 0.7, '
                            '"severity": "none", "taint_source": "hardcoded literal", '
                            '"explanation": "mock fallback response"}'),
            }

        response = {
            "id": "chatcmpl-mock", "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # quiet


def start_mock_server(port: int = 11500, respond_with_tool_call: bool = True):
    server = http.server.HTTPServer(("localhost", port), _Handler)
    server.respond_with_tool_call = respond_with_tool_call  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    srv = start_mock_server()
    print("Mock OpenAI-compatible server running on http://localhost:11500/v1")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
