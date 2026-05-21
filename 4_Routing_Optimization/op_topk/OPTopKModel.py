"""
OPTopKModel.py
===============
POMO attention model for the Orienteering Problem (OP).

Encoder input:   (x, y, prize)  -- 3 features, Linear(3, embedding_dim)
Decoder context: [last_node_embedding | remaining_budget]
                  embedding_dim          1

Follows AM paper (Kool et al. 2019) Appendix D for OP:
    - Prize fed as encoder input feature (eq. 24)
    - Remaining budget fed to decoder context (eq. 26)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OPTopKModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params  = model_params
        self.encoder       = OP_Encoder(**model_params)
        self.decoder       = OP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        """Run encoder once per episode — called before the decoding loop."""
        # node_xy_prize: (batch, top_k, 3)
        self.encoded_nodes = self.encoder(reset_state.node_xy_prize)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        B = state.BATCH_IDX.size(0)
        P = state.BATCH_IDX.size(1)
        device = self.encoded_nodes.device

        if state.current_node is None:
            selected = torch.arange(P, device=device)[None, :].expand(B, P)
            prob = torch.ones(B, P, device=device)
            encoded_first = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q_first(encoded_first)

        else:
            encoded_last = _get_encoding(self.encoded_nodes, state.current_node)

            # IMPORTANT:
            # Finished rows have all -inf in ninf_mask.
            # If we send all -inf into softmax, it becomes NaN.
            safe_ninf_mask = state.ninf_mask.clone()
            if state.finished is not None:
                safe_ninf_mask[state.finished] = 0.0

            probs = self.decoder(
                encoded_last,
                state.remaining_budget,
                ninf_mask=safe_ninf_mask,
            )

            # Give finished routes a dummy valid action
            if state.finished is not None:
                probs = probs.clone()
                probs[state.finished] = 0.0
                probs[state.finished, 0] = 1.0

            if self.training or self.model_params['eval_type'] == 'softmax':
                selected = (
                    probs.reshape(B * P, -1)
                        .multinomial(1)
                        .squeeze(1)
                        .reshape(B, P)
                )
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index):
    B, P  = node_index.shape
    D     = encoded_nodes.size(2)
    idx   = node_index[:, :, None].expand(B, P, D)
    return encoded_nodes.gather(dim=1, index=idx)


# ─────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────

class OP_Encoder(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim     = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        # 3 input features: x, y, prize  (key change from TSP's 2)
        self.embedding = nn.Linear(3, embedding_dim)
        self.layers    = nn.ModuleList([
            EncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])

    def forward(self, node_xy_prize):
        out = self.embedding(node_xy_prize)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.head_num = model_params['head_num']
        embedding_dim = model_params['embedding_dim']
        qkv_dim       = model_params['qkv_dim']

        self.Wq      = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.Wk      = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.Wv      = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(self.head_num * qkv_dim, embedding_dim)
        self.norm1   = AddAndInstanceNorm(**model_params)
        self.ff      = FeedForward(**model_params)
        self.norm2   = AddAndInstanceNorm(**model_params)

    def forward(self, x):
        q   = reshape_by_heads(self.Wq(x), self.head_num)
        k   = reshape_by_heads(self.Wk(x), self.head_num)
        v   = reshape_by_heads(self.Wv(x), self.head_num)
        out = mha(q, k, v)
        out = self.combine(out)
        out = self.norm1(x, out)
        out = self.norm2(out, self.ff(out))
        return out


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class OP_Decoder(nn.Module):
    """
    Context at each step t:
        q = Wq_first(h_first) + Wq_last([h_last | remaining_budget])

    h_last tells the model WHERE it is.
    remaining_budget tells it HOW CONSTRAINED it is.
    h_first is the POMO trajectory anchor (which starting node).

    Together these let the model learn:
        "High budget + near high-prize node  → go there"
        "Low budget  + far high-prize node   → skip it"
    """

    def __init__(self, **model_params):
        super().__init__()
        self.head_num = model_params['head_num']
        self.sqrt_emb = model_params['sqrt_embedding_dim']
        self.clip     = model_params['logit_clipping']
        embedding_dim = model_params['embedding_dim']
        qkv_dim       = model_params['qkv_dim']

        # [last_node_emb | remaining_budget] → query
        self.Wq_last  = nn.Linear(embedding_dim + 1, self.head_num * qkv_dim, bias=False)
        # first_node_emb → query anchor
        self.Wq_first = nn.Linear(embedding_dim,     self.head_num * qkv_dim, bias=False)

        self.Wk      = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.Wv      = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(self.head_num * qkv_dim, embedding_dim)

        self.k               = None
        self.v               = None
        self.single_head_key = None
        self.q_first         = None

    def set_kv(self, encoded_nodes):
        self.k               = reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        self.v               = reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q_first(self, encoded_first):
        self.q_first = reshape_by_heads(self.Wq_first(encoded_first), self.head_num)

    def forward(self, encoded_last_node, remaining_budget, ninf_mask):
        # Build context: where + how constrained
        context = torch.cat(
            [encoded_last_node, remaining_budget[:, :, None]], dim=2
        )   # (B, P, embedding_dim+1)

        q_last = reshape_by_heads(self.Wq_last(context), self.head_num)
        q      = self.q_first + q_last

        out  = mha(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out  = self.combine(out)

        score        = torch.matmul(out, self.single_head_key)
        score_scaled = score / self.sqrt_emb
        score_clip   = self.clip * torch.tanh(score_scaled)
        score_masked = score_clip + ninf_mask

        return F.softmax(score_masked, dim=2)


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def reshape_by_heads(qkv, head_num):
    B, N, _ = qkv.shape
    return qkv.reshape(B, N, head_num, -1).transpose(1, 2)


def mha(q, k, v, rank3_ninf_mask=None):
    B, H, N, D = q.shape
    K = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3)) / (D ** 0.5)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :].expand(B, H, N, K)
    weights = F.softmax(score, dim=3)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(B, N, H * D)


class AddAndInstanceNorm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = nn.InstanceNorm1d(
            model_params['embedding_dim'], affine=True, track_running_stats=False
        )

    def forward(self, x, residual):
        return self.norm((x + residual).transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.W1 = nn.Linear(model_params['embedding_dim'], model_params['ff_hidden_dim'])
        self.W2 = nn.Linear(model_params['ff_hidden_dim'], model_params['embedding_dim'])

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))
