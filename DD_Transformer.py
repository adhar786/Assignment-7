import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DecoderLayer(nn.Module):
    # d_model stands for dimension of the word vector in a model
    # drop out defines a dropout rate for regularization;
    # scr_attention? Source-target attention, i.e. encoder decoder attention
    # d_ff, feed-forward dimension, is the dimension of inner layer of the Feedforward
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_heads, dropout)
        self.src_attention = MultiheadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        # Initiate three different Normal Layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    # tgt_mask? target mask, i.e. the musked attention in decoder
    # src_mask? Source mask, it is used to eliminate padding from the encoder part
    def forward(self, x, memory, src_mask, tgt_mask):
        # Self-Attention
        # Residual is used for add layer after the norm
        residual = x
        x = self.layer_norm1(x)
        # What type of mask is tgt_mask; target mask used for masked attention
        # x, _ means _ is a placeholder for attentionweights, which is not
        # important in this context
        x, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout(x)
        x += residual

        # Source-Target Attention
        residual = x
        x = self.layer_norm2(x)
        # ? Cross attention with encoder, memory is from encoder representations
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
        # What is assert; assert makes sure d_model can be divided by n_heads
        # Otherwise, it will raise an attribute error
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        # what is nn.Linear; initiate a matrix
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
        # ? ; view methods reshaped the tensor to a 4D tensor, -1 makes it automatically
        # calculate the sequence length, transpose swapped the second and third positions
        # i.e. the number of heads with the length of the sequence
        query = query.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        # ? transpose the last two dimensions of the tensor
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        # Apply softmax function to the last dimension of the scores, i.e. the sequence length
        # ? still confused why dim = -1 not -2
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, value)

        # Merge heads
        # ? contiguous ensures memory is contiguous during transpose
        # we swap the number of heads and length of sequence back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head)

        # Linear transformation
        # ? times another linear transformation
        output = self.linear_out(context)

        return output, attention_weights

# d_ff stands for diffuse, similar to a dense layer
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
        # Use the linear layer to transform embedded vector back to word
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    # trg? Stands for the target that is used during teacher forcing
    # memory? The vector representation from encoder
    # takes in a tensor with 2D shape for trg, memory as 3D tensor,
    def forward(self, trg, memory, src_mask, trg_mask):
        trg = self.embedding(trg) * math.sqrt(self.d_model)  # Scale embedding
        trg = self.dropout(trg)

        for layer in self.layers:
            trg = layer(trg, memory, src_mask, trg_mask)

        # F.log_softmax? we use log_softmax for numerical stability
        output = F.log_softmax(self.linear(trg), dim=-1)
        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.heads, self.head_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Self-attention layer
        self.attention = SelfAttention(embed_size, heads)
        # Normalization layer 1
        self.norm1 = nn.LayerNorm(embed_size)
        # Feedforward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        # Normalization layer 2
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply self-attention
        attention = self.attention(x)
        # Add & normalize (residual connection)
        x = self.norm1(x + attention)
        # Apply feedforward layers
        forward = self.feed_forward(x)
        # Add & normalize (residual connection)
        out = self.norm2(x + forward)
        out = self.dropout(out)
        return out


class Transformer(nn.Module):
    def __init__(self, d_model, heads, vocab_size, forward_expansion, dropout=0.1, n_layers=1):
        super(Transformer, self).__init__()
        self.encoder = TransformerBlock(d_model, heads, forward_expansion, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_layers, heads, forward_expansion, dropout)

    # The forward function returns the original input from encoder and the next word id from vocab dictionary
    # as a 2d tensor
    def forward(self, src_input, trg_input, trg_mask, scr_mask=None):
        memory = self.encoder(src_input)
        log_probs = self.decoder(trg_input, memory, scr_mask, trg_mask)
        probs = torch.exp(log_probs[:, :, :])
        next_ids = probs.argmax(dim=-1)
        return next_ids


# <editor-fold desc="sample code">
d_model = 100
heads = 2
vocab_size = 10000
forward_expansion = 1

transformer = Transformer(d_model, heads, vocab_size, forward_expansion)
trg_input = torch.tensor([[1,2,3]]).long()
src_input = torch.randn(1,3,100)
trg_mask = torch.tensor([[1, 1, 1]])
print(transformer(src_input, trg_mask, trg_input))
# </editor-fold>
d_model = 100
heads = 1
vocab_size = 7620
forward_expansion = 1

transformer = Transformer(d_model, heads, vocab_size, forward_expansion)
# We plug 3 because tensor cannot have negative number during embedding
trg_input = torch.tensor(np.load('decoder_target.npy')[:-1, :] + 3).long()
np_input = np.load('encoder_input.npy')
src_input = torch.tensor(np_input, dtype=torch.float32).transpose(0, 2).transpose(1, 2)
trg_mask = torch.tensor(np.ones(10), dtype=torch.float32)
print(f'shape of trg_input is {trg_input.shape}')
print(f'shape of src_input is {src_input.shape}')
print(transformer(src_input, trg_input, trg_mask))
