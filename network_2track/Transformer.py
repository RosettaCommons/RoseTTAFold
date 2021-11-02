import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from performer_pytorch import SelfAttention

# Functions for Transformer architecture
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Own implementation for multihead attention (Input shape: Batch, Len, Emb)
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch, L = query.shape[:2]
        q = self.to_query(query).view(batch, L, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        k = self.to_key(key).view(batch, L, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        v = self.to_value(value).view(batch, L, self.heads, self.d_k).permute(0,2,1,3)
        #
        attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.d_k)
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        attention = self.dropout(attention)
        #
        out = torch.matmul(attention, v) # (B, h, L, d_k)
        out = out.permute(0,2,1,3).contiguous().view(batch, L, -1)
        #
        out = self.to_out(out)
        return out


# Use PreLayerNorm for more stable training
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=None):
        super(EncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        # multihead attention
        if self.use_performer:
            self.attn = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                      nb_features=64, generalized_attention=True, **performer_opts)
        else:
            self.attn = MultiheadAttention(d_model, heads, dropout=p_drop)
        # feedforward
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model)

        # normalization module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, src):
        # Input shape for multihead attention: (BATCH, SRCLEN, EMB)
        # multihead attention w/ pre-LayerNorm
        B, N, L = src.shape[:3]
        src2 = self.norm1(src)
        src2 = src2.reshape(B*N, L, -1)
        src2 = self.attn(src2, src2, src2).reshape(B,N,L,-1)
        src = src + self.dropout1(src2)

        # feed-forward
        src2 = self.norm2(src) # pre-normalization
        src2 = self.linear2(self.dropout(F.relu_(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layer, d_model):
        super(Encoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
        #self.norm = nn.LayerNorm(d_model)
   
    def forward(self, src):

        output = src
        for layer in self.layers:
            output = layer(output)
        return output

class InterEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, p_drop=0.1):
        super(InterEncoderLayer, self).__init__()
        # multihead attention
        self.attn = MultiheadAttention(d_model, heads, k_dim=d_k, v_dim=d_v, dropout=p_drop)
        # feedforward
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model)

        # normalization module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, src, tgt):
        # Input:
        #   For MSA to Pair: src (N, L, K), tgt (L, L, C)
        #   For Pair to MSA: src (L, L, C), tgt (N, L, K)
        # Input shape for multihead attention: (SRCLEN, BATCH, EMB)
        # multihead attention
        # pre-normalization
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(tgt2, src, src) # projection to query, key, value are done in MultiheadAttention module
        tgt = tgt + self.dropout1(tgt2)

        G# feed-forward
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu_(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class InterEncoder(nn.Module):
    def __init__(self, enc_layer, n_layer, d_model):
        super(InterEncoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
        #self.norm = nn.LayerNorm(d_model)
    def forward(self, src, tgt):
        output = tgt
        for layer in self.layers:
            output, att = layer(src, output)
        return output ,att
        #return self.norm(output), att

class SpecialEncoderLayer(nn.Module):
    def __init__(self, heads, d_in, d_out, d_ff, p_drop=0.1):
        super(SpecialEncoderLayer, self).__init__()
        self.heads = heads
        
        # linear projection to get attention map
        self.norm = nn.LayerNorm(d_in)
        self.proj_pair_1 = nn.Linear(d_in, heads//2)
        self.proj_pair_2 = nn.Linear(d_in, heads//2)
        # linear projection to get values from given msa
        self.proj_msa = nn.Linear(d_out, d_out)
        # projection after applying attention
        self.proj_out = nn.Linear(d_out, d_out)
        # dropouts
        self.drop_1 = nn.Dropout(p_drop)
        self.drop_2 = nn.Dropout(p_drop)
        self.drop_3 = nn.Dropout(p_drop)
        # feed-forward
        self.linear1 = nn.Linear(d_out, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_out)
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)

    def forward(self, src, tgt):
        # Input:
        #  For pair to msa: src=pair (B, L, L, C), tgt=msa (B, N, L, K)
        B, N, L = tgt.shape[:3]
        # get attention map
        src = self.norm(src)
        attn_map_1 = F.softmax(self.proj_pair_1(src), dim=1).permute(0,3,1,2) # (B, h//2, L, L)
        attn_map_2 = F.softmax(self.proj_pair_2(src), dim=2).permute(0,3,2,1) # (B, h//2, L, L)
        attn_map = torch.cat((attn_map_1, attn_map_2), dim=1) # (B, h, L, L)
        attn_map = self.drop_1(attn_map).unsqueeze(1)
        
        # apply attention
        tgt2 = self.norm1(tgt.view(B*N,L,-1)).view(B,N,L,-1)
        value = self.proj_msa(tgt2).permute(0,3,1,2).contiguous().view(B, -1, self.heads, N, L) # (B,-1, h, N, L)
        tgt2 = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0,2,3,1) # (B,N,L,K)
        tgt2 = self.proj_out(tgt2)
        tgt = tgt + self.drop_2(tgt2)

        # feed-forward
        tgt2 = self.norm2(tgt.view(B*N,L,-1)).view(B,N,L,-1)
        tgt2 = self.linear2(self.dropout(F.relu_(self.linear1(tgt2))))
        tgt = tgt + self.drop_3(tgt2)

        return tgt

class SpecialEncoder(nn.Module):
    def __init__(self, enc_layer, n_layer, d_model):
        super(SpecialEncoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
        #self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, tgt):

        output = tgt
        for layer in self.layers:
            output= layer(src, output)
        return output

