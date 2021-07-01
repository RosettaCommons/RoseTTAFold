import csv
from dateutil import parser
import numpy as np
import torch
from torch.utils import data
import os
base_dir = "/projects/ml/TrRosetta/PDB30-20FEB17"
base_torch_dir = base_dir
if not os.path.exists(base_dir):
    # training on blue
    base_dir = "/gscratch/TrRosetta/"
    if os.path.exists("/scratch/torch/hhr"):
        base_torch_dir = "/scratch"
    else:
        base_torch_dir = base_dir

def set_data_loader_params(args):
    PARAMS = {
        "LIST"    : "%s/list.no_denovo.csv"%base_dir,
        "VAL"     : "%s/val_lists/xaa"%base_dir,
        "DIR"     : base_torch_dir,
        "MINTPLT" : 1,
        "MAXTPLT" : 10,
        "MINSEQ"  : 1,
        "MAXSEQ"  : 500,
        "LMIN"    : 100,
        "LMAX"    : 150,
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : 3.5,
        "SLICE"   : "CONT",
        "ROWS"    : 1,
        "SUBSMP"  : "UNI",
        "seqID"   : 50.0,
        "MAXTOKEN": 2**15
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

def msa_collate_fn(batch, params):
    # To deal with various depth/length inputs
    # Inputs:
    #  - batch: [msa, xyz, idx]
    #  - msa: raw MSA before subsampling/cropping (N, L)
    #  - xyz: raw backbone coordinates before cropping (L, 3, 3)
    #  - idx: raw residue index from pdb before cropping (L)
    # Outputs:
    #  - batched msa (B, N', L'), xyz (B, L', 3, 3), idx (B,L')

    msa_s, xyz_s, idx_s, seqlen, depth = zip(*[[msa, xyz, idx, msa.shape[1], msa.shape[0]] for msa, xyz, idx in batch])
    L = min(seqlen)
    device = msa_s[0].device

    # 1. Slice long chains
    if L > params['LMAX']:
        L = np.random.randint(params['LMIN'], params['LMAX'])
    
    # 2. set how many sequences in subsampled MSA (nmin~nmax)
    nmin = params['MINSEQ']
    nmax = min(depth)
    nmax = min(nmax, params['MAXTOKEN']//L)
    nmax = min(nmax, params['MAXSEQ'])
    n_sample = subsample_msa(nmin, nmax, params['SUBSMP'])


    b_msa = list()
    b_xyz = list()
    b_idx = list()
    for i_b in range(len(msa_s)):
        tmp_msa = msa_s[i_b]
        tmp_xyz = xyz_s[i_b]
        tmp_idx = idx_s[i_b]
        #
        if depth[i_b] > n_sample + 1: # should be subsampled
            if n_sample > 0:
                # n_sample unique indices
                if n_sample > 50:
                    sample = torch.randperm(depth[i_b]-1, device=device)[:n_sample]
                else:
                    sample = torch.randperm(min(depth[i_b]-1, 100), device=device)[:n_sample]
                tmp_msa = torch.cat([msa_s[i_b][:1,:], msa_s[i_b][1:,:][sample]], dim=0)
            else:
                tmp_msa = msa_s[i_b][:1,:]
        if seqlen[i_b] > L: # trim inputs to size L
            sel = get_crop(tmp_msa, L, params)
            #
            tmp_msa = tmp_msa[:,sel]
            tmp_xyz = xyz_s[i_b][sel]
            tmp_idx = idx_s[i_b][sel]
        #
        b_msa.append(tmp_msa)
        b_xyz.append(tmp_xyz)
        b_idx.append(tmp_idx)
    #
    b_msa = torch.stack(b_msa, 0)
    b_xyz = torch.stack(b_xyz, 0)
    b_idx = torch.stack(b_idx, 0)

    return b_msa, b_xyz, b_idx

def msa_infer_collate_fn(batch, params):
    # To deal with various depth/length inputs
    # Inputs:
    #  - batch: [msa, xyz, idx]
    #  - msa: raw MSA before subsampling/cropping (N, L)
    #  - xyz: raw backbone coordinates before cropping (L, 3, 3)
    #  - idx: raw residue index from pdb before cropping (L)
    # Outputs:
    #  - batched msa (B, N', L'), xyz (B, L', 3, 3), idx (B,L')

    msa_s, xyz_s, idx_s, seqlen, depth = zip(*[[msa, xyz, idx, msa.shape[1], msa.shape[0]] for msa, xyz, idx in batch])
    L = min(seqlen)
    device = msa_s[0].device
    
    # 1. Slice long chains
    if L > params['LMAX']:
        L = np.random.randint(params['LMIN'], params['LMAX'])

    # 2. set how many sequences in subsampled MSA (nmin~nmax)
    nmin = params['MINSEQ']
    nmax = min(depth)
    nmax = min(nmax, params['MAXTOKEN']//L)
    nmax = min(nmax, params['MAXSEQ'])
    n_sample = subsample_msa(nmin, nmax, 'CONST')

    b_msa = list()
    b_xyz = list()
    b_idx = list()
    for i_b in range(len(msa_s)):
        tmp_msa = msa_s[i_b]
        tmp_xyz = xyz_s[i_b]
        tmp_idx = idx_s[i_b]
        #
        if depth[i_b] > n_sample + 1: # should be subsampled
            if n_sample > 0:
                tmp_msa = torch.cat([msa_s[i_b][:1,:], msa_s[i_b][1:n_sample+1,:]], dim=0)
            else:
                tmp_msa = msa_s[i_b][:1,:]
        if seqlen[i_b] > L: # trim inputs to size L
            sel = get_crop(tmp_msa, L, params)
            #
            tmp_msa = tmp_msa[:,sel]
            tmp_xyz = xyz_s[i_b][sel]
            tmp_idx = idx_s[i_b][sel]
        #
        b_msa.append(tmp_msa)
        b_xyz.append(tmp_xyz)
        b_idx.append(tmp_idx)
    #
    b_msa = torch.stack(b_msa, 0)
    b_xyz = torch.stack(b_xyz, 0)
    b_idx = torch.stack(b_idx, 0)

    return b_msa, b_xyz, b_idx

def tbm_collate_fn(batch, params):
    # To deal with various depth/length inputs
    # Inputs:
    #  - batch: [msa, xyz, idx, xyz_t, t1d]
    #  - msa: raw MSA before subsampling/cropping (N, L)
    #  - xyz: raw backbone coordinates before cropping (L, 3, 3)
    #  - idx: raw residue index from pdb before cropping (L)
    #  - xyz_t: raw backbone coordinates of templates before cropping (T, L, 3, 3)
    #  - t1d: raw 1D template features before cropping (T, L, 3)
    # Outputs:
    #  - batched msa (B, N', L'), xyz (B, L', 3, 3), idx (B,L')

    msa_s, xyz_s, idx_s, xyz_t_s, t1d_s, t0d_s, seqlen, depth, tplt_s = zip(*[[msa, xyz, idx, xyz_t, t1d, t0d, msa.shape[1], msa.shape[0], xyz_t.shape[0]] for msa, xyz, idx, xyz_t, t1d, t0d in batch])
    L = min(seqlen)
    ntplt = min(tplt_s)
    device = msa_s[0].device
    
    # 1. Slice long chains
    if L > params['LMAX']:
        L = np.random.randint(params['LMIN'], params['LMAX'])

    # 2. set how many sequences in subsampled MSA (nmin~nmax)
    nmin = params['MINSEQ']
    nmax = min(depth)
    nmax = min(nmax, params['MAXTOKEN']//L)
    nmax = min(nmax, params['MAXSEQ'])
    n_sample = subsample_msa(nmin, nmax, params['SUBSMP'])

    # 3. Pick templates
    # pick a random number of random templates
    nmin = min(ntplt,params['MINTPLT'])
    nmax = min(ntplt,params['MAXTPLT']) + 1
    prob = np.array([max(10.0-2.0*i, 1.0) for i in range(nmin, nmax)])
    prob = prob / prob.sum()
    npick = np.random.choice(np.arange(nmin, nmax), p=prob)

    b_msa = list()
    b_xyz = list()
    b_idx = list()
    b_xyz_t = list()
    b_t1d = list()
    b_t0d = list()
    for i_b in range(len(msa_s)):
        tmp_msa = msa_s[i_b]
        tmp_xyz = xyz_s[i_b]
        tmp_idx = idx_s[i_b]
        if npick > 0:
            tmp_xyz_t = xyz_t_s[i_b][:npick]
            tmp_t1d = t1d_s[i_b][:npick]
            tmp_t0d = t0d_s[i_b][:npick]
        else:
            tmp_xyz_t = torch.full((1,seqlen[i_b],3,3), np.nan).float()
            tmp_t1d = torch.zeros((1, seqlen[i_b], 3)).float()
            tmp_t0d = torch.zeros((1,3)).float()
        #
        if depth[i_b] > n_sample + 1: # should be subsampled
            if n_sample > 0:
                # n_sample unique indices
                if n_sample > 50:
                    sample = torch.randperm(depth[i_b]-1, device=device)[:n_sample]
                else:
                    sample = torch.randperm(min(depth[i_b]-1, 100), device=device)[:n_sample]
                tmp_msa = torch.cat([msa_s[i_b][:1,:], msa_s[i_b][1:,:][sample]], dim=0)
            else:
                tmp_msa = msa_s[i_b][:1,:]
        if seqlen[i_b] > L: # trim inputs to size L
            sel = get_crop(tmp_msa, L, params)
            #
            tmp_msa = tmp_msa[:,sel]
            tmp_xyz = xyz_s[i_b][sel]
            tmp_idx = idx_s[i_b][sel]
            tmp_xyz_t = tmp_xyz_t[:,sel]
            tmp_t1d = tmp_t1d[:, sel]
        #
        b_msa.append(tmp_msa)
        b_xyz.append(tmp_xyz)
        b_idx.append(tmp_idx)
        b_xyz_t.append(tmp_xyz_t)
        b_t1d.append(tmp_t1d)
        b_t0d.append(tmp_t0d)
    #
    b_msa = torch.stack(b_msa, 0)
    b_xyz = torch.stack(b_xyz, 0)
    b_idx = torch.stack(b_idx, 0)
    b_xyz_t = torch.stack(b_xyz_t, 0)
    b_t1d = torch.stack(b_t1d, 0)
    b_t0d = torch.stack(b_t0d, 0)

    return b_msa, b_xyz, b_idx, b_xyz_t, b_t1d, b_t0d

def tbm_infer_collate_fn(batch, params):
    # To deal with various depth/length inputs
    # Inputs:
    #  - batch: [msa, xyz, idx, xyz_t, t1d]
    #  - msa: raw MSA before subsampling/cropping (N, L)
    #  - xyz: raw backbone coordinates before cropping (L, 3, 3)
    #  - idx: raw residue index from pdb before cropping (L)
    #  - xyz_t: raw backbone coordinates of templates before cropping (T, L, 3, 3)
    #  - t1d: raw 1D template features before cropping (T, L, 3)
    # Outputs:
    #  - batched msa (B, N', L'), xyz (B, L', 3, 3), idx (B,L')

    msa_s, xyz_s, idx_s, xyz_t_s, t1d_s, t0d_s, seqlen, depth, tplt_s = zip(*[[msa, xyz, idx, xyz_t, t1d, t0d, msa.shape[1], msa.shape[0], xyz_t.shape[0]] for msa, xyz, idx, xyz_t, t1d, t0d in batch])
    L = min(seqlen)
    ntplt = min(tplt_s)
    device = msa_s[0].device

    # 1. Slice long chains
    if L > params['LMAX']:
        L = np.random.randint(params['LMIN'], params['LMAX'])

    # 2. set how many sequences in subsampled MSA (nmin~nmax)
    nmin = params['MINSEQ']
    nmax = min(depth)
    nmax = min(nmax, params['MAXTOKEN']//L)
    nmax = min(nmax, params['MAXSEQ'])
    n_sample = subsample_msa(nmin, nmax, 'CONST')

    # 3. Pick templates
    # pick a random number of random templates
    npick = min(ntplt,params['MAXTPLT'])

    b_msa = list()
    b_xyz = list()
    b_idx = list()
    b_xyz_t = list()
    b_t1d = list()
    b_t0d = list()
    for i_b in range(len(msa_s)):
        tmp_msa = msa_s[i_b]
        tmp_xyz = xyz_s[i_b]
        tmp_idx = idx_s[i_b]
        tmp_xyz_t = xyz_t_s[i_b][:npick]
        tmp_t1d = t1d_s[i_b][:npick]
        #
        if depth[i_b] > n_sample + 1: # should be subsampled
            if n_sample > 0:
                tmp_msa = torch.cat([msa_s[i_b][:1,:], msa_s[i_b][1:n_sample+1,:]], dim=0)
            else:
                tmp_msa = msa_s[i_b][:1,:]
        if seqlen[i_b] > L: # trim inputs to size L
            sel = get_crop(tmp_msa, L, params)
            #
            tmp_msa = tmp_msa[:,sel]
            tmp_xyz = xyz_s[i_b][sel]
            tmp_idx = idx_s[i_b][sel]
            tmp_xyz_t = tmp_xyz_t[:,sel]
            tmp_t1d = tmp_t1d[:, sel]
        #
        b_msa.append(tmp_msa)
        b_xyz.append(tmp_xyz)
        b_idx.append(tmp_idx)
        b_xyz_t.append(tmp_xyz_t)
        b_t1d.append(tmp_t1d)
        b_t0d.append(t0d_s[i_b][:npick])
    #
    b_msa = torch.stack(b_msa, 0)
    b_xyz = torch.stack(b_xyz, 0)
    b_idx = torch.stack(b_idx, 0)
    b_xyz_t = torch.stack(b_xyz_t, 0)
    b_t1d = torch.stack(b_t1d, 0)
    b_t0d = torch.stack(b_t0d, 0)

    return b_msa, b_xyz, b_idx, b_xyz_t, b_t1d, b_t0d

def subsample_msa(nmin, nmax, method):
    # how many sequences to pick?
    if method == 'LOG':
        nlog = np.random.uniform(low=np.log(nmin),high=np.log(nmax))
        n = np.exp(nlog).round().astype(int)-1
    elif method == 'UNI':
        n = np.random.uniform(nmin,nmax)
        n = np.round(n).astype(int)-1
    elif method == "CONST":
        n = nmax-1
    else:
        sys.exit('Error: wrong MSA subsampling mode:', params['SUBSMP'])

    if n <= 0:
        n = 0
    return n

# slice long chains
def get_crop(msa, size, params):

    device = msa.device
    l = msa.shape[1]
    sel = torch.arange(l,device=device)
    
    if params['SLICE'] == 'CONT':
        # slice continuously
        start = np.random.randint(l-size+1)
        return sel[start:start+size]

    elif params['SLICE'] == 'DISCONT':
        # slice discontinuously
        size1 = np.random.randint(params['LMIN']//2, size//2)
        size2 = size - size1
        gap = np.random.randint(l-size+1)

        start1 = np.random.randint(l-size1-size2-gap+1)
        sel1 = sel[start1:start1+size1]
        sel2 = sel[start1+size1+gap:start1+size1+size2+gap]

        return torch.cat([sel1,sel2])

    else:
        sys.exit('Error: wrong cropping mode:', params['SLICE'])

    return sel

def pick_templates(tplt, qlen, params, pick_top=False, seqID_cut=80.0):
    if seqID_cut <= 100.0:
        sel = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[sel]
        tplt['qmap'] = tplt['qmap'][:,sel]
        tplt['xyz'] = tplt['xyz'][:, sel]
        tplt['f1d'] = tplt['f1d'][:, sel]
        tplt['f0d'] = tplt['f0d'][:, sel]
    
    ntplt = len(tplt['ids'])
    if ntplt<1:
        # xyz coords and positional scores
        return torch.full((1,qlen,3,3),np.nan).float(),torch.zeros((1,qlen,3)).float(),torch.zeros((1,3)).float()
   
    if not pick_top:
        ## pick a random number of random templates
        #nmin = min(ntplt,params['MINTPLT'])
        #nmax = min(ntplt,params['MAXTPLT']) + 1
        #npick = np.random.randint(nmin,nmax)
        npick = min(ntplt, params['MAXTPLT']*5)
        sample = torch.randperm(ntplt)[:npick]
    else:
        npick = min(ntplt, params['MAXTPLT'])
        sample = torch.arange(npick)

    xyz = torch.full((npick,qlen,3,3),np.nan).float()
    f1d = torch.zeros((npick,qlen,3)).float()
    f0d = list()

    for i,nt in enumerate(sample):
        sel = torch.where(tplt['qmap'][0,:,1]==nt)[0]
        pos = tplt['qmap'][0,sel,0]
        xyz[i,pos] = tplt['xyz'][0,sel,:3]
        f1d[i,pos] = tplt['f1d'][0,sel,:3]
        # 0-D features: HHprob, seqID, similarity
        f0d.append(torch.stack([tplt['f0d'][0,nt,0]/100.0, tplt['f0d'][0,nt,4]/100.0, tplt['f0d'][0,nt,5]], dim=-1))

    return xyz,f1d,torch.stack(f0d, dim=0)

def get_train_valid_set(params):
    # read validation IDs
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])

    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    # compile training and validation sets
    rows_subset = rows[::params['ROWS']]
    train = {}
    valid = {}
    for r in rows_subset:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    return train, valid


def loader_tbm(item, params, pick_top=False, seqID=80.0):

    pdb = torch.load(params['DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = torch.load(params['DIR']+'/torch/a3m/'+item[1][2:4]+'/'+item[1]+'.pt')
    tplt = torch.load(params['DIR']+'/torch/hhr/'+item[1][2:4]+'/'+item[1]+'.pt')
    idx = pdb['idx'][0]
    
    l = a3m['msa'].shape[-1]
    xyz_t,f1d_t,f0d_t = pick_templates(tplt, l, params, pick_top=pick_top, seqID_cut=seqID)
    
    # trim to physically observed residues
    msa = a3m['msa'][...,idx]
    ins = a3m['ins'][...,idx]
    xyz_t = xyz_t[:,idx]
    f1d_t = f1d_t[:,idx]

    return msa[0], pdb['xyz'][0,:,:3], idx, xyz_t, f1d_t, f0d_t

def loader_msa(item, params, pick_top=False):

    pdb = torch.load(params['DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = torch.load(params['DIR']+'/torch/a3m/'+item[1][2:4]+'/'+item[1]+'.pt')
    idx = pdb['idx'][0]
    
    # trim to physically observed residues
    msa = a3m['msa'][...,idx]

    return msa[0], pdb['xyz'][0,:,:3], idx

class Dataset(data.Dataset):
    def __init__(self, IDs, loader, train_dict, params, pick_top=False):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params, pick_top=self.pick_top)
        return out

