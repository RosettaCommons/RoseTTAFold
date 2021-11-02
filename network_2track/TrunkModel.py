import torch
import torch.nn as nn
from Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from Attention_module import IterativeFeatureExtractor
from DistancePredictor import DistanceNetwork
from InitStrGenerator import InitStr_Network

class TrunkModule(nn.Module):
    def __init__(self, n_module=4, n_diff_module=2, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, d_attn=50, d_crd=64,
                 r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 use_templ=False):
        super(TrunkModule, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, p_drop=p_drop)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_diff_module=n_diff_module,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts)
        self.c6d_predictor = DistanceNetwork(1, d_pair, block_type='bottle', p_drop=p_drop)
        # TODO
        self.crd_predictor = InitStr_Network(d_model=d_pair, d_hidden=d_hidden, 
                                             d_attn=d_attn, d_out=d_crd, d_msa=d_msa,
                                             n_layers=n_layer, n_att_head=n_head_msa, 
                                             p_drop=p_drop,
                                             performer_opts=performer_L_opts)

    def forward(self, msa, seq, idx, t1d=None, t2d=None):
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)
        #
        # Extract features
        msa, pair = self.feat_extractor(msa, pair)
        # Predict 3D coordinates of CA atoms # TODO
        crds = self.crd_predictor(msa, pair, seq, idx)

        # Predict 6D coords
        pair = pair.view(B, L, L, -1).permute(0,3,1,2) # (B, C, L, L) 
        logits = self.c6d_predictor(pair.contiguous())
        #return logits
        return logits, crds.view(B,L,3,3) # TODO
