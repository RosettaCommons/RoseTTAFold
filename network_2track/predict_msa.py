import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m
from TrunkModel  import TrunkModule
import util
from collections import namedtuple

script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

MODEL_PARAM ={
        "n_module"     : 12,
        "n_diff_module": 12,
        "n_layer"      : 1,
        "d_msa"        : 64,
        "d_pair"       : 128,
        "d_templ"      : 128,
        "n_head_msa"   : 4,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "d_attn"       : 64,
        "d_crd"        : 64,
        "r_ff"         : 2,
        "n_resblock"   : 1,
        "p_drop"       : 0.1,
        "performer_N_opts": {},
        "performer_L_opts": {}
        }

class Predictor():
    def __init__(self, model_dir=None, use_cpu=False):
        if model_dir == None:
            self.model_dir = "%s/weights"%(script_dir)
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "RF2t"
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = TrunkModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name):
        chk_fn = "%s/%s.pt"%(self.model_dir, model_name)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True
    
    def predict(self, a3m_fn, npz_fn, L1):
        msa = parse_a3m(a3m_fn)
        N, L = msa.shape
       
        self.model.eval()
        with torch.no_grad():
            #
            msa = torch.tensor(msa[:10000], device=self.device).long().view(1, -1, L)
            idx_pdb = torch.arange(L, device=self.device).long().view(1, L)
            idx_pdb[:,L1:] += 200 
            seq = msa[:,0]
            #
            logit_s, _ = self.model(msa, seq, idx_pdb)
            
            # distogram
            prob = self.active_fn(logit_s[0])
            prob = prob.permute(0,2,3,1).cpu().detach().numpy()
            # interchain contact prob
            prob = np.sum(prob.reshape(L,L,-1)[:L1,L1:,:20], axis=-1).astype(np.float16)
        
        # store inter-chain contact prob only
        np.savez_compressed(npz_fn, dist=prob)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_dir", default="%s/weights"%(script_dir),
                        help="Path to pre-trained network weights [%s/weights]"%script_dir)
    parser.add_argument("-msa", required=True, help="Input paired MSA file")
    parser.add_argument("-npz", required=True, help="output npz file name")
    parser.add_argument("-L1", required=True, type=int, help="Length of first chain")
    parser.add_argument("--cpu", dest='use_cpu', default=False, action='store_true')
    args = parser.parse_args()

    pred = Predictor(model_dir=args.model_dir, use_cpu=args.use_cpu)
    pred.predict(args.msa, args.npz, args.L1)
