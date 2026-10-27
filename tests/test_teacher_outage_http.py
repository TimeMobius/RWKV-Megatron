"""Real refused TCP connections recover, or exhaust a bounded HTTP window."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from rwkv_megatron.online_teacher import OnlineTeacherClient


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        payload = json.dumps({'choices': [{'prompt_logprobs': [None, {'2': {'logprob': -0.1}}]}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_actual_connection_refusal_then_service_restart_succeeds():
    reserved = socket.socket()
    reserved.bind(('127.0.0.1', 0))
    port = reserved.getsockname()[1]
    client = OnlineTeacherClient(f'http://127.0.0.1:{port}', 'fixture', '/unused',
                                top_k=1, max_retries=0, retry_window=2,
                                retry_backoff_initial=0.02, retry_backoff_max=0.05,
                                retry_jitter=0, api_key='')
    session = client._get_session()
    session.trust_env = False
    post = session.post
    server = thread = None

    def restarting_post(*args, **kwargs):
        nonlocal server, thread
        try:
            return post(*args, **kwargs)
        except requests.ConnectionError:
            assert server is None
            reserved.close()
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            raise

    session.post = restarting_post
    try:
        ids, _, metrics = client._request_prompt_logprobs('hello', 2, return_metrics=True)
        assert ids[1, 0] == 2
        assert metrics['retries'] == 1 and client.get_stats()['failures'] == 0
    finally:
        reserved.close()
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(2)


def test_actual_persistent_connection_refusal_exits_after_window():
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        client = OnlineTeacherClient(f'http://127.0.0.1:{reserved.getsockname()[1]}',
                                    'fixture', '/unused', top_k=1, retry_window=0.1,
                                    retry_backoff_initial=0.02, retry_backoff_max=0.04,
                                    retry_jitter=0, api_key='')
        client._get_session().trust_env = False
        started = time.monotonic()
        with pytest.raises(TimeoutError, match='deadline exhausted'):
            client._request_prompt_logprobs('hello', 2)
        assert 0.09 <= time.monotonic() - started < 2
        assert client.get_stats()['retries'] >= 2
        assert client.get_stats()['failures'] == 1
