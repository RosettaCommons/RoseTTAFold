#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import multiprocessing as mp

script_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

def smooth(x, window_len=13, window='hanning'):
    s = np.r_[[x[0]]*(window_len//2), x, [x[-1]]*(window_len//2)]
    if window == 'flat': #moving average
        w = np.onew(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def lrQres2CAdev(estogram,mask,minseqsep=13):
    # first get masked contact list
    nres = len(estogram)
    contacts = []
    for i in range(nres):
        for j in range(nres):
            if abs(i-j) < minseqsep: continue ## up to 3 H turns
            if mask[i][j] > 0.1: contacts.append((i,j,mask[i][j]))

    lddt_raw = [0.0 for i in range(nres)]
    Psum = [0.0001 for i in range(nres)]
    for (i,j,P) in contacts:
        in05 = estogram[i][j][7]
        in1 = np.sum(estogram[i][j][6:9])
        in2 = np.sum(estogram[i][j][5:10])
        in4 = np.sum(estogram[i][j][4:11])
        inall = P*(in05+in1+in2+in4)/4.0
        lddt_raw[i] += inall
        lddt_raw[j] += inall
        Psum[i] += P
        Psum[j] += P
        
    lddt_lr = np.array([lddt_raw[i]/Psum[i] for i in range(nres)])
    lddt_lr = smooth(lddt_lr)
    # v1:
    #CAdev = [np.exp(5*(0.6-lddt_res)) for lddt_res in lddt_lr]
    # Qlr 0.8 -> err 0.4 Ang
    # Qlr 0.7 -> err 0.6 Ang
    # Qlr 0.6 -> err 1.0 Ang
    # Qlr 0.5 -> err 1.6 Ang
    # Qlr 0.4 -> err 2.7 Ang
    # Qlr 0.3 -> err 4.5 Ang
    # Qlr 0.2 -> err 7.4 Ang
    
    # v2:
    CAdev = [1.5*np.exp(4*(0.7-lddt_res)) for lddt_res in lddt_lr]
    # Qlr 0.8 -> err 1.0 Ang
    # Qlr 0.7 -> err 1.0 Ang
    # Qlr 0.6 -> err 2.2 Ang
    # Qlr 0.5 -> err 3.3 Ang
    # Qlr 0.4 -> err 5.0 Ang
    # Qlr 0.3 -> err 7.4 Ang
    # Qlr 0.2 -> err 11.  Ang
    
    return CAdev

def calc_lddt_dist(args):
    i, j, pose_s = args
    pose_i = pose_s[i]
    pose_j = pose_s[j]
    #
    lddt_1 = float(os.popen("%s/lddt/lddt -c %s %s | grep Glob"%(script_dir, pose_i, pose_j)).readlines()[-1].split()[-1])
    lddt_2 = float(os.popen("%s/lddt/lddt -c %s %s | grep Glob"%(script_dir, pose_j, pose_i)).readlines()[-1].split()[-1])
    lddt = (lddt_1 + lddt_2) / 2.0
    return 1 - lddt

infolder = sys.argv[1]
outfolder = sys.argv[2]
n_core = int(sys.argv[3])

if not os.path.exists(outfolder):
    os.mkdir(outfolder)

pdb_s = glob.glob("%s/model*.pdb"%infolder)
pdb_s.sort()

lddt_s = list()
for pdb in pdb_s:
    npz_fn = pdb[:-4] + ".npz"
    if not os.path.exists(npz_fn):
        continue
    lddt = np.load(npz_fn)['lddt']
    lddt_s.append((np.mean(lddt), os.path.abspath(pdb)))

# clustering top50% structures from trRefine
lddt_s.sort(reverse=True)
selected = list()
score_s = list()
n_str = len(lddt_s) // 2
for i in range(n_str):
    pdb = lddt_s[i][-1]
    selected.append(pdb)
    score_s.append(lddt_s[i][0])
score_s = np.array(score_s)
selected = np.array(selected)

args = list()
for i in range(n_str-1):
    for j in range(i+1, n_str):
        args.append((i,j,selected))

n_core_pool = min(n_core, len(args))
pool = mp.Pool(n_core_pool)
raw_dist = pool.map(calc_lddt_dist, args)
pool.close()
pool.join()
dist = np.zeros((n_str, n_str), dtype=np.float)
idx = np.triu_indices(n_str, k=1)
dist[idx] = raw_dist
dist = dist + dist.T
#
cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average').fit(dist)
#
unique_labels = np.unique(cluster.labels_)
rep_s = list()
for label in unique_labels:
    idx = np.where(cluster.labels_==label)[0]
    #
    Emin_idx = np.argmax(score_s[idx])
    model = selected[idx][Emin_idx]
    rep_s.append((score_s[idx][Emin_idx], os.path.abspath(model)))
rep_s.sort(reverse=True)

i_model = 0
modelQ_dat = list()
for score, pdb in rep_s:
    i_model += 1
    os.system("ln -sf %s %s/model_%d.pdb"%(pdb, outfolder, i_model))
    #
    dat = np.load(pdb[:-4] + ".npz")
    esto = dat['estogram'].astype(np.float32) + 1e-9
    mask = dat['mask'].astype(np.float32) + 1e-9
    #esto = esto.transpose(1,2,0)
    #
    CAdev = lrQres2CAdev(esto, mask)
    #
    wrt = ''
    with open(pdb) as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            resNo = int(line[22:26])
            wrt += line[:60] + " %5.2f"%CAdev[resNo-1] + line[66:]
    with open("%s/model_%d.crderr.pdb"%(outfolder, i_model), 'wt') as fp:
        fp.write(wrt)
    #
    modelQ_dat.append("model_%d     %.4f   %s"%(i_model, score, os.path.realpath(pdb)))

with open("%s/modelQ.dat"%outfolder, 'wt') as fp:
    fp.write("\n".join(modelQ_dat))
