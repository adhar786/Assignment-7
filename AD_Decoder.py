import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_heads, dropout)
        self.src_attention = MultiheadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Self-Attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout(x)
        x += residual

        # Source-Target Attention
        residual = x
        x = self.layer_norm2(x)
        x, _ = self.src_attention(x, memory, memory, src_mask)
        x = self.dropout(x)
        x += residual

        # Feed Forward
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x += residual

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Split into multiple heads
        query = query.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, value)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head)

        # Linear transformation
        output = self.linear_out(context)

        return output, attention_weights

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, trg, memory, src_mask, trg_mask):
        trg = self.embedding(trg) * math.sqrt(self.d_model)  # Scale embedding
        trg = self.dropout(trg)

        for layer in self.layers:
            trg = layer(trg, memory, src_mask, trg_mask)

        output = F.log_softmax(self.linear(trg), dim=-1)
        return output
