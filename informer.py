import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
    def forward(self, queries, keys, values):
        attn_output, _ = self.attention(queries, keys, values)
        return attn_output

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=4, dropout=0.1):
        super().__init__()
        self.attention = StandardAttention(d_model, n_heads).to(device)
        self.conv1 = nn.Conv1d(d_model, d_ff, 3, padding=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Conv1d for distilling
        print(x.transpose(1, 2).shape)
        print(self.conv1)
        conv_out = self.conv2(F.gelu(self.conv1(x.permute(0, 2, 1)))).transpose(1, 2)
        x = self.norm2(x + self.dropout(conv_out))
        return x[:, ::2, :]  # Halve sequence length

class Informer(nn.Module):
    # def __init__(self, K, seq_len=96, d_model=512, n_heads=8, e_layers=2, d_layers=1):
    def __init__(self, aggregate_mat, params):
        super().__init__()
                 
        input_channels = params['n_series']
        K = params['n_series']
        seq_len = params['context_window']
        d_model = params['hidden_size']
        
        e_layers=2
        d_layers=1
        n_heads = 8
        self.params = params

                 
        self.encoder_embedding = nn.Linear(K, d_model)
        self.decoder_embedding = nn.Linear(K, d_model)
        self.encoder_layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads).to(device)
                                           for _ in range(e_layers)])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True), 
            num_layers=d_layers
        )
        self.projection = nn.Linear(d_model, K)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        if self.params.get('project', False):
            self.projector = Projection(aggregate_mat)

                 
    def forward(self, src, tgt=None):
        # src: (B, seq_len, K)
        src = src.permute(0, 2, 1)
        B = src.shape[0]
        
        # Encoder
        enc_out = self.encoder_embedding(src) + self.pos_encoder
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # Decoder (generative inference)
        if tgt is None:  # Use start token during inference
            start_token = torch.zeros(B, 1, src.shape[-1], device=src.device)
            dec_out = self.decoder_embedding(start_token)
        else:
            dec_out = self.decoder_embedding(tgt)
            
        dec_out = self.decoder(
            dec_out, 
            enc_out,
            tgt_mask=nn.Transformer().generate_square_subsequent_mask(dec_out.size(1)).to(src.device)
        )
        output = self.projection(dec_out[:, -1:, :])  # Predict 1 time horizon
        
        if self.params.get('project', False):
            return self.projector.project(output)
        
        return output