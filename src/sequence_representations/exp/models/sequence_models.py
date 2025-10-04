"""Sequence model implementations: RNN, LSTM, GRU for text classification.

Includes:
- RNNClassifier: unified interface for RNN/LSTM/GRU
- Packed sequence handling for variable-length inputs
- Gate introspection for LSTM/GRU analysis
- Bidirectional support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities ----------

def pack_padded(x, lengths):
    """Pack padded sequences for efficient RNN processing."""
    # x: (B, T), lengths: (B,)
    lengths_sorted, idx = lengths.sort(descending=True)
    x_sorted = x.index_select(0, idx)
    return nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True), idx

def unpack_padded(packed, idx):
    """Unpack sequences and restore original order."""
    x_unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    # restore order
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device)
    return x_unpacked.index_select(0, inv)


class RNNClassifier(nn.Module):
    """Unified RNN/LSTM/GRU classifier for text classification."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, bidirectional=False, rnn_type='rnn', dropout=0.1, num_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=1)  # 1 is <pad>
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        RNN = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = RNN(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)

        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size*mult, num_classes)
        )

    def forward(self, x, lengths):
        """Forward pass with packed sequences."""
        # x: (B, T)
        e = self.emb(x)  # (B, T, E)

        packed, idx = pack_padded(x, lengths)
        # BUT we need the embeddings in the same sorted order
        e_sorted = e.index_select(0, idx)
        packed_e = nn.utils.rnn.pack_padded_sequence(e_sorted, lengths[idx].cpu(), batch_first=True, enforce_sorted=True)

        H_all_packed, H_last = self.rnn(packed_e)

        # Get final hidden state(s)
        if self.rnn_type == 'lstm':
            h_T, c_T = H_last
            h_use = h_T
        else:
            h_use = H_last

        # h_use: (num_layers*D, B, H)
        if self.bidirectional:
            # concat last layer's forward/backward states
            h_last_layer = h_use.view(self.rnn.num_layers, (2), h_use.size(1), h_use.size(2))[-1]
            h_cat = torch.cat([h_last_layer[0], h_last_layer[1]], dim=-1)  # (B, 2H)
        else:
            h_cat = h_use[-1]  # (B, H)

        # Restore order to original batch
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(idx.numel(), device=idx.device)
        h_cat = h_cat.index_select(0, inv)

        logits = self.head(h_cat)
        return logits


def build_model(cfg, vocab_size):
    """Build sequence model from config."""
    return RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=cfg.get("embed_dim", 128),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
        rnn_type=cfg.get("type", "lstm"),
        dropout=cfg.get("dropout", 0.1),
        num_classes=cfg.get("num_classes", 4),
    )


# -------- Introspection (gate probe) --------
@torch.no_grad()
def gate_probe_batch(model, x, lengths):
    """Compute gate statistics for LSTM/GRU introspection.
    
    Returns dict of scalar means to log for TensorBoard.
    """
    stats = {}
    if not isinstance(model.rnn, (nn.LSTM, nn.GRU)):
        return stats

    # Embed
    e = model.emb(x)  # (B,T,E)

    # Sort by length, prepare packed embeddings
    lengths_sorted, idx = lengths.sort(descending=True)
    e = e.index_select(0, idx)

    # We'll only look at the first layer's weights
    if isinstance(model.rnn, nn.LSTM):
        W_ih = model.rnn.weight_ih_l0  # (4H, E)
        W_hh = model.rnn.weight_hh_l0  # (4H, H)
        b_ih = model.rnn.bias_ih_l0
        b_hh = model.rnn.bias_hh_l0
        H = model.rnn.hidden_size

        h = torch.zeros(e.size(0), H, device=e.device)
        c = torch.zeros(e.size(0), H, device=e.device)
        f_vals, i_vals, o_vals, hn_vals = [], [], [], []

        for t in range(e.size(1)):
            x_t = e[:, t, :]
            gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            i, f, g, o = gates.split(H, dim=-1)
            i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            f_vals.append(f.mean()); i_vals.append(i.mean()); o_vals.append(o.mean()); hn_vals.append(h.norm(dim=-1).mean())

        stats["gates/forget_mean"] = torch.stack(f_vals).mean().item()
        stats["gates/input_mean"]  = torch.stack(i_vals).mean().item()
        stats["gates/output_mean"] = torch.stack(o_vals).mean().item()
        stats["stats/hidden_norm"] = torch.stack(hn_vals).mean().item()

    elif isinstance(model.rnn, nn.GRU):
        W_ih = model.rnn.weight_ih_l0  # (3H, E)
        W_hh = model.rnn.weight_hh_l0  # (3H, H)
        b_ih = model.rnn.bias_ih_l0
        b_hh = model.rnn.bias_hh_l0
        H = model.rnn.hidden_size

        h = torch.zeros(e.size(0), H, device=e.device)
        z_vals, r_vals, hn_vals = [], [], []

        for t in range(e.size(1)):
            x_t = e[:, t, :]
            gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            z, r, n = gates.split(H, dim=-1)
            z = torch.sigmoid(z); r = torch.sigmoid(r)
            n = torch.tanh(n)  # approximation (ignores r-hh coupling detail)
            h = (1 - z) * h + z * n
            z_vals.append(z.mean()); r_vals.append(r.mean()); hn_vals.append(h.norm(dim=-1).mean())

        stats["gates/update_mean"] = torch.stack(z_vals).mean().item()
        stats["gates/reset_mean"]  = torch.stack(r_vals).mean().item()
        stats["stats/hidden_norm"] = torch.stack(hn_vals).mean().item()

    return stats
