import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Transformer import EncoderLayer, Encoder

# Initial embeddings for target sequence, msa, template info
# positional encoding
#   option 1: using sin/cos --> using this for now 
#   option 2: learn positional embedding

class PositionalEncodeing(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=5000):
        super(PositionalEncodeing, self).__init__()
        self.drop = nn.Dropout(p_drop)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # (1, max_len, d_model)
    def forward(self, x, idx_s):
        pe = list()
        for idx in idx_s:
            pe.append(self.pe[:,idx,:])
        pe = torch.stack(pe)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class MSA_emb(nn.Module):
    def __init__(self, d_model=64, d_msa=21, p_drop=0.1, max_len=5000):
        super(MSA_emb, self).__init__()
        self.emb = nn.Embedding(d_msa, d_model)
        self.pos = PositionalEncodeing(d_model, p_drop=p_drop, max_len=max_len)
    def forward(self, msa, idx):
        B, N, L = msa.shape
        out = self.emb(msa) # (B, N, L, K//2)
        out = self.pos(out, idx) # add positional encoding
        return out

# pixel-wise attention based embedding (from trRosetta-tbm)
class Templ_emb(nn.Module):
    def __init__(self, d_t1d=3, d_t2d=10, d_templ=64, n_att_head=4, r_ff=4, p_drop=0.1):
        super(Templ_emb, self).__init__()
        self.proj = nn.Linear(d_t1d*2+d_t2d+1, d_templ)
        # attention along T
        enc_layer = EncoderLayer(d_model=d_templ, d_ff=d_templ*r_ff,
                                 heads=n_att_head, p_drop=p_drop)
        self.encoder = Encoder(enc_layer, 1, d_templ)
        #
        # dimension reduction (B,T,L,L,-1) -> (B, L, L, -1)
        self.to_v = nn.Linear(d_templ, d_templ)
        self.to_u = nn.Linear(d_templ, 1, bias=False)

    def forward(self, t1d, t2d, idx):
        # Input
        #   - t1d: 1D template info (B, T, L, 2)
        #   - t2d: 2D template info (B, T, L, L, 10)
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).repeat(1,1,1,L,1)
        right = t1d.unsqueeze(2).repeat(1,1,L,1,1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1 
        seqsep = torch.log(seqsep.float()).view(B,L,L,1).unsqueeze(1).repeat(1,T,1,1,1)
        #
        feat = torch.cat((t2d, left, right, seqsep), -1)
        feat = self.proj(feat) # (B, T, L, L, d_templ)
        #
        # attention along T
        feat = feat.permute(0,2,3,1,4).contiguous().view(B,L*L, T, -1)
        feat = self.encoder(feat).view(B*L*L, T, -1)

        # dimension reduction using attention
        v = torch.tanh(self.to_v(feat)) # (B*L*L, T, A)
        vu = self.to_u(v).view(B*L*L, T)
        alphas = F.softmax(vu, dim=-1).view(B*L*L, T, 1) # attention map

        feat = torch.matmul(alphas.transpose(-2,-1), feat).view(B,L,L,-1)
        return feat

class Pair_emb_w_templ(nn.Module):
    def __init__(self, d_model=128, d_seq=21, d_templ=64):
        super(Pair_emb_w_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + d_templ + 1, d_model)
    def forward(self, seq, idx, templ):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).repeat(1,1,L,1)
        right = seq.unsqueeze(1).repeat(1,L,1,1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1 
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)
        #
        pair = torch.cat((left, right, templ, seqsep), dim=-1)
        pair = self.projection(pair)
        return pair

class Pair_emb_wo_templ(nn.Module):
    #TODO: embedding without template info
    def __init__(self, d_model=128, d_seq=21):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model + 1, d_model)
    def forward(self, seq, idx):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).repeat(1,1,L,1)
        right = seq.unsqueeze(1).repeat(1,L,1,1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1 
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)
        #
        pair = torch.cat((left, right, seqsep), dim=-1)
        pair = self.projection(pair)
        return pair

