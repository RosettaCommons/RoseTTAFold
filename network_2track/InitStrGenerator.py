# Take pair features, predict initial CA coordinates (Lx3) or (N,CA,C) coordinates (Lx9)
# + playing with RMSD loss
# architecture from Minkyung's SS predictor (how to reduce pair features to node features)
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Encoder, EncoderLayer

def get_seqsep(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:,None,:] - idx[:,:,None]
    sign = torch.sign(seqsep)
    seqsep = sign * torch.log(torch.abs(seqsep) + 1.0)
    return seqsep.unsqueeze(-1)

def get_tiled_1d_features(seq, node=None):
    '''
    Input:
        - seq: target sequence in integer (B,L)
    Output:
        - tiled 1d features including 1hot encoded sequence (B, L, 21) and node features if given
    '''
    B,L = seq.shape
    #
    seq1hot = F.one_hot(seq, num_classes=21).float()
    if node != None:
        feat_1d = torch.cat((seq1hot, node), dim=-1)
    else:
        feat_1d = seq1hot

    left = feat_1d.view(B,L,1,-1).expand(-1,-1,L,-1)
    right = feat_1d.view(B,1,L,-1).expand(-1,L,-1,-1)
    return torch.cat((left, right), dim=-1) # (B, L, L, -1)

class Attention(nn.Module):
    def __init__(self, d_model=128, d_attn=50):
        super(Attention, self).__init__()
        #
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_u = nn.Linear(d_attn, 1, bias=False)
    
    def forward(self, x, time_major=False):
        if time_major:
            L, BL = x.shape[:2]
            x = x.permute(1,0,2) # make it as (Batch, Time, Feats)
        else:
            BL, L = x.shape[:2]
        v = torch.tanh(self.to_v(x)) # (B, T, A)
        vu = self.to_u(v).view(BL, L) # (B, T)
        alphas = F.softmax(vu, dim=-1).view(BL,L,1)

        x = torch.matmul(alphas.transpose(-2,-1), x).view(BL, -1)

        return x


class InitStr_Network(nn.Module):
    def __init__(self, d_model=128, d_hidden=64, d_out=64, d_attn=50, d_msa=64,
                 n_rnn_layer=2, n_layers=2, n_att_head=4, r_ff=2, p_drop=0.1,
                 performer_opts=None):
        super(InitStr_Network, self).__init__()
        self.norm_node = nn.LayerNorm(d_msa)
        self.norm_edge = nn.LayerNorm(d_model)
        #
        self.proj_mix = nn.Linear(d_model+d_msa*2+21*2+1, d_hidden)
        
        enc_layer_1 = EncoderLayer(d_model=d_hidden, d_ff=d_hidden*r_ff, heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layers, d_hidden)  
        
        enc_layer_2 = EncoderLayer(d_model=d_hidden, d_ff=d_hidden*r_ff, heads=n_att_head, p_drop=p_drop,
                                 performer_opts=performer_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layers, d_hidden)  
        
        self.attn = Attention(d_model=d_hidden*2, d_attn=d_attn)
        self.proj = nn.Linear(d_hidden*2, d_out)
        
        enc_layer_3 = EncoderLayer(d_model=d_out, d_ff=d_out*r_ff, heads=n_att_head, p_drop=p_drop,
                                 performer_opts=performer_opts)
        self.encoder_3 = Encoder(enc_layer_3, n_layers, d_out)  

        self.proj_crd = nn.Linear(d_out, 9) # predict BB coordinates
        
    def forward(self, msa, pair, seq, idx):
        B, L = pair.shape[:2]
        msa = self.norm_node(msa)
        pair = self.norm_edge(pair)
        #
        node_feats = msa.mean(1) # (B, L, K)
        #
        tiled_1d = get_tiled_1d_features(seq, node=node_feats)
        seqsep = get_seqsep(idx)
        pair = torch.cat((pair, seqsep, tiled_1d), dim=-1)
        pair = self.proj_mix(pair)

        # reduce dimension
        hidden_1 = self.encoder_1(pair).view(B*L, L, -1) # (B*L, L, d_hidden)
        #
        pair = pair.view(B, L, L, -1).permute(0,2,1,3)
        hidden_2 = self.encoder_2(pair).reshape(B*L, L, -1) # (B*L, L, d_hidden)
        pair = torch.cat((hidden_1, hidden_2), dim=-1)
        out = self.attn(pair) # (B*L, d_hidden)
        out = self.proj(out).view(B, L, -1) # (B, L, d_out)
        #
        out = self.encoder_3(out.reshape(B, 1, L, -1))
        xyz = self.proj_crd(out).view(B,L,3,3) # (B, L, 3, 3)

        return xyz

