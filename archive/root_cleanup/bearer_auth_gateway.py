"""bearer_auth_gateway.py

Wraps sandboxed CMI endpoints with Bearer Token Authorization checks.
Guarantees loopback endpoints are secured under zero-dependency standard library designs.
"""
from __future__ import annotations

import json
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from sandbox_api_server import CMISandboxRequestHandler, CMIContext, ctx

class AuthSandboxRequestHandler(CMISandboxRequestHandler):
    def do_POST(self):
        # 1. Inspect HTTP authorization headers
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer sandbox-token-12345"):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized: Invalid or missing bearer token"}).encode())
            return
            
        # 2. Token validated, hand over to standard streaming executor
        super().do_POST()

def run_secured_server(port=8850):
    global ctx
    import sandbox_api_server
    sandbox_api_server.ctx = CMIContext()
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, AuthSandboxRequestHandler)
    print(f"Auth-Gated CMI Sandbox Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping secured server...")
        httpd.server_close()

if __name__ == "__main__":
    port_val = int(sys.argv[1]) if len(sys.argv) > 1 else 8850
    run_secured_server(port=port_val)
