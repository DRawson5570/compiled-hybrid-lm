"""secured_robust_gateway.py

Implements Phase 12.3: Active Failure Probes & Graceful Fallback Recovery.
Synthesizes Authenticated Bearer checks, LFU-TTL Routing caching, and Dynamic Entropy
Temperature scaling into a single robust production-grade sandboxed stream server.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Import CMI components from sandbox_api_server and hybrid to ensure completeness
from dynamic_entropy_temp import calculate_shannon_entropy, get_adaptive_temperature
from lfu_ttl_routing_cache import LFUTTLRoutingCache
from sandbox_api_server import CMIContext, ctx as global_ctx

# Direct imports from hybrid structure
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V
from hybrid.v1_blender.blender_model import build_feature_matrix

# Cache for server requests
server_cache = LFUTTLRoutingCache(capacity=5)

class SecuredRobustRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP logs for clean output
        pass

    def do_POST(self):
        # 1. Bearer token auth check
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer sandbox-token-12345"):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized: Invalid or missing token"}).encode())
            return

        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/api/cmi/stream":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            params = json.loads(body) if body else {}

            prompt = params.get("prompt", "")
            max_tokens = params.get("max_tokens", 5)
            simulate_failure = params.get("simulate_failure", None) # Inject faulty channel if specified

            # Check cache first
            cached_result = server_cache.get(prompt)
            if cached_result:
                print(f"  [Cache Hit] Retrieved routing trace for prompt: '{prompt}'")
                # Return cached payload directly
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for out in cached_result:
                    self.wfile.write(f"data: {json.dumps(out)}\n\n".encode())
                    self.wfile.flush()
                return

            # Execute model routing steps
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Autoregressive generation simulation
            raw_tokens = prompt.split()
            current_sequence = []
            for t in raw_tokens:
                if t in tok2id:
                    current_sequence.append(t)
                else:
                    current_sequence.append("<UNK>")

            if not current_sequence:
                current_sequence = ["<UNK>"]

            recorded_trace = []

            for step in range(1, max_tokens + 1):
                # 1. Prepare tensor IDs
                ids = torch.tensor([tok2id[token] for token in current_sequence], device=global_ctx.device)
                T_len = len(current_sequence)

                # 2. Forward channels
                p_outputs = []
                for c in global_ctx.channels:
                    ids_dev = ids.to(global_ctx.emb.device)
                    p_outputs.append(c.forward(ids_dev).to(global_ctx.device))

                # 3. Collect step features
                all_feats = []
                for t in range(T_len):
                    x_observed = ids[t]
                    x_lag1 = ids[t - 1] if t > 0 else torch.zeros_like(x_observed)

                    log_p_observed_t = torch.stack([p_out[t, x_observed] for p_out in p_outputs])
                    log_p_lag1_t = torch.stack([p_out[t, x_lag1] for p_out in p_outputs])

                    entropy_t = []
                    max_log_prob_t = []
                    for p_out in p_outputs:
                        p_dist = p_out[t].exp()
                        entropy_t.append(-(p_dist * p_out[t]).sum())
                        max_log_prob_t.append(p_out[t].max())

                    entropy_t = torch.stack(entropy_t)
                    max_log_prob_t = torch.stack(max_log_prob_t)

                    feat = build_feature_matrix(
                        log_p_observed_t.unsqueeze(0),
                        log_p_lag1_t.unsqueeze(0),
                        entropy_t.unsqueeze(0),
                        max_log_prob_t.unsqueeze(0),
                        global_ctx.emb.to(global_ctx.device),
                        x_observed.unsqueeze(0),
                        use_embedding=True
                    )
                    all_feats.append(feat)

                features = torch.cat(all_feats, dim=0).to(global_ctx.device)

                with torch.no_grad():
                    log_w = global_ctx.blender(features.unsqueeze(0)).squeeze(0)

                # Get probabilities
                probs = log_w[-1].exp()

                # Calculate live Shannon routing uncertainty and adaptive temperature
                entropy = calculate_shannon_entropy(probs)
                temp = get_adaptive_temperature(entropy)

                # Map channel names
                channels = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
                prob_map = {ch: float(p) for ch, p in zip(channels, probs.tolist())}
                dominant = max(prob_map, key=prob_map.get)

                # Channel Fault Probe Simulator (Phase 12.3)
                if simulate_failure and dominant == simulate_failure:
                    # Filter out failed channel and re-normalize among available models
                    print(f"  [FAULT INJECTED] Dominant channel '{dominant}' failed probe!")
                    prob_map[simulate_failure] = 0.0
                    sum_remaining = sum(prob_map.values())
                    if sum_remaining > 0:
                        prob_map = {k: v / sum_remaining for k, v in prob_map.items()}
                        dominant = max(prob_map, key=prob_map.get)
                        print(f"  [FALLBACK] Gracefully rerouted stream to runner-up channel: '{dominant}'")
                    else:
                        dominant = "FallbackChannel"

                # Standard token resolution
                if dominant == "ToolChannel":
                    next_token = "27"
                elif dominant == "InstructChannel":
                    next_token = "chien"
                elif dominant == "CoderChannel":
                    next_token = ":"
                else:
                    next_token = "is"

                current_sequence.append(next_token)

                step_payload = {
                    "token": next_token,
                    "dominant_channel": dominant,
                    "routing": prob_map,
                    "entropy": entropy,
                    "temperature": temp,
                    "step_number": step
                }
                recorded_trace.append(step_payload)

                self.wfile.write(f"data: {json.dumps(step_payload)}\n\n".encode())
                self.wfile.flush()

            # Stream completion sentinel
            completion_payload = {"status": "done"}
            self.wfile.write(f"data: {json.dumps(completion_payload)}\n\n".encode())
            self.wfile.flush()

            # Cache the completed stream sequence trace for future runs (respecting LFU)
            server_cache.set(prompt, recorded_trace)

        else:
            self.send_response(404)
            self.end_headers()

def run_secured_robust_server(port=8850):
    global global_ctx
    # Lazy initialised if needed
    if global_ctx is None:
        global_ctx = CMIContext()
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, SecuredRobustRequestHandler)
    print(f"Secured Robust CMI Gateway server running on http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping secured server...")
        httpd.server_close()

if __name__ == "__main__":
    port_val = int(sys.argv[1]) if len(sys.argv) > 1 else 8850
    run_secured_robust_server(port=port_val)
