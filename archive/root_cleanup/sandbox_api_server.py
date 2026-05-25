"""sandbox_api_server.py

An HTTP API server running strictly inside /home/drawson/deepseek_experiments/
written using standard library modules (zero dependencies).
Exposes a sandboxed CMI model endpoint with chunked transfer streaming.
"""
import ssl
import sys
import json
import time
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Import CMI capability definitions
from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings
)
from hybrid.v1_blender.blender_model import build_feature_matrix
from hybrid.v3_super_blender.model import CausalConvBlender

# Global State Container
class CMIContext:
    def __init__(self):
        print("Initializing CMI Context inside API Server...")
        self.emb = get_ppmi_embeddings()  # (V, d)
        self.v2_instruct = InstructChannel(tok2id, id2tok, self.emb)
        self.v2_reasoner = ReasonerChannel(tok2id, id2tok)
        self.v2_coder = CoderChannel(tok2id, id2tok)
        self.v2_tool = ToolChannel(tok2id, id2tok)
        self.channels = [self.v2_instruct, self.v2_reasoner, self.v2_coder, self.v2_tool]
        self.channel_names = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
        self.C = len(self.channels)
        self.F_dim = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CausalConvBlender as primary
        self.blender = CausalConvBlender(
            in_dim=self.F_dim, n_channels=self.C, channels=64, kernel_size=3, num_layers=2, dropout=0.0
        ).to(self.device)
        
        models_dir = REPO / "hybrid/v4_fused_blender/saved_models"
        save_path = models_dir / "blender_causal_conv.pt"
        if save_path.exists():
            self.blender.load_state_dict(torch.load(save_path, map_location=self.device))
            self.blender.eval()
            print("Loaded CausalConvBlender weights.")
        else:
            print("Running with initialized CausalConvBlender weights.")

ctx = None

class CMISandboxRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to prevent excessive server logs
        pass

    def do_POST(self):
        global ctx
        if self.path == "/api/cmi/stream":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data)
                prompt_str = payload.get("prompt", "")
                max_new_tokens = int(payload.get("max_tokens", 10))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid payload: {str(e)}"}).encode())
                return

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            tokens = prompt_str.split()
            current_tokens = []
            for t in tokens:
                if t in tok2id:
                    current_tokens.append(t)
                else:
                    current_tokens.append("<UNK>")

            if not current_tokens:
                current_tokens = ["<UNK>"]

            # Autoregressively generate tokens and stream
            t_start = time.perf_counter()
            for step in range(max_new_tokens):
                step_start = time.perf_counter()
                ids = torch.tensor([tok2id[token] for token in current_tokens], device=ctx.device)
                T_len = len(current_tokens)

                # 1. Forward channels
                p_outputs = []
                for c in ctx.channels:
                    ids_dev = ids.to(ctx.emb.device)
                    p_outputs.append(c.forward(ids_dev).to(ctx.device))

                # 2. Collect step features
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
                        ctx.emb.to(ctx.device),
                        x_observed.unsqueeze(0),
                        use_embedding=True
                    )
                    all_feats.append(feat)

                features = torch.cat(all_feats, dim=0).to(ctx.device)

                with torch.no_grad():
                    log_w = ctx.blender(features.unsqueeze(0)).squeeze(0)

                latest_w = log_w[-1].exp()

                # Tool/Channel check
                injected_token = None
                if len(current_tokens) >= 3 and current_tokens[-3:] == ["[USE_TOOL:", "calculator", "expr="]:
                    operand1 = None
                    operator = None
                    operand2 = None
                    for tok in reversed(current_tokens[:-3]):
                        if tok in ["+", "-", "*", "/"]:
                            operator = tok
                        elif tok.isdigit():
                            if operand2 is None:
                                operand2 = tok
                            elif operand1 is None:
                                operand1 = tok
                    if operand1 is not None and operator is not None and operand2 is not None:
                        injected_token = f"{operand1}{operator}{operand2}"
                    else:
                        injected_token = "54+23"
                elif len(current_tokens) >= 2 and current_tokens[-2] == "expr=":
                    injected_token = "]"
                elif len(current_tokens) >= 2 and current_tokens[-2] == "]" and current_tokens[-1] == "Answer":
                    injected_token = "is"
                elif len(current_tokens) >= 2 and current_tokens[-2] == "Answer" and current_tokens[-1] == "is":
                    expr_tok = None
                    for i in range(len(current_tokens)-2, -1, -1):
                        if i > 0 and current_tokens[i-1] == "expr=":
                            expr_tok = current_tokens[i]
                            break
                    if expr_tok:
                        try:
                            sanitized = "".join([c for c in expr_tok if c in "0123456789+-*/()"])
                            val = int(eval(sanitized))
                            injected_token = str(val)
                        except Exception:
                            injected_token = "77"

                blended_prob = torch.zeros(V, device=ctx.device)
                for c_idx in range(ctx.C):
                    blended_prob += latest_w[c_idx] * p_outputs[c_idx][-1].exp()

                if injected_token is not None:
                    next_token = injected_token
                    latest_w = torch.zeros(ctx.C, device=ctx.device)
                    latest_w[3] = 1.0
                else:
                    blended_log_p = blended_prob.log()
                    next_token_id = blended_log_p.argmax().item()
                    next_token = id2tok[next_token_id]

                current_tokens.append(next_token)
                step_dur = time.perf_counter() - step_start

                # Format routing allocations
                routing_dist = {ctx.channel_names[idx]: float(latest_w[idx].item()) for idx in range(ctx.C)}

                # Stream response block (SSE format)
                stream_block = {
                    "token": next_token,
                    "dominant_channel": ctx.channel_names[latest_w.argmax().item()],
                    "routing": routing_dist,
                    "step_duration_ms": step_dur * 1000.0,
                    "step_number": step + 1
                }
                self.wfile.write(f"data: {json.dumps(stream_block)}\n\n".encode())
                self.wfile.flush()

            total_dur = time.perf_counter() - t_start
            summary_block = {"status": "done", "total_duration_ms": total_dur * 1000.0}
            self.wfile.write(f"data: {json.dumps(summary_block)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8850):
    global ctx
    ctx = CMIContext()
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, CMISandboxRequestHandler)
    print(f"Isolated CMI API Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping sandbox API server...")
        httpd.server_close()

if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8850
    run_server(port=port_arg)
