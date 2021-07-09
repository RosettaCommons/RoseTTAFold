import sys
import os
import numpy as np

a3m1 = sys.argv[1]
a3m2 = sys.argv[2]

def read_a3m(fn):
    '''parse an a3m files as a dictionary {label->sequence}'''
    seq = []
    lab = []
    is_first = True
    for line in open(fn, "r"):
        if line[0] == '>':
            label = line.strip()[1:]
            is_incl = True
            if is_first: # include first sequence (query)
                is_first = False
                lab.append(label)
                continue
            if "UniRef" in label:
                code = label.split()[0].split('_')[-1]
                if code.startswith("UPI"): # UniParc identifier -- exclude
                    is_incl = False
                    continue
            elif label.startswith("tr|"):
                code = label.split('|')[1]
            else:
                is_incl = False
                continue
            lab.append(code)
        else:
            if is_incl:
                seq.append(line.rstrip())
            else:
                continue

    return seq, lab

# https://www.uniprot.org/help/accession_numbers
def uni2idx(ids):
    '''convert uniprot ids into integers according to the structure
    of uniprot accession numbers'''
    ids2 = [i+'AAA0' if len(i)==6 else i for i in ids]
    arr = np.array([list(s) for s in ids2], dtype='|S1').view(np.uint8)

    for i in [1,5,9]:
        arr[:,i] -= ord('0')

    arr[arr>=ord('A')] -= ord('A')
    arr[arr>=ord('0')] -= ord('0')-26

    arr[:,0][arr[:,0]>ord('Q')-ord('A')] -= 3

    arr = arr.astype(np.int64)

    coef = np.array([23,10,26,36,36,10,26,36,36,1], dtype=np.int64)
    coef = np.tile(coef[None,:],[len(ids),1])

    c1 = [i for i,id_ in enumerate(ids) if id_[0] in 'OPQ' and len(id_)==6]
    c2 = [i for i,id_ in enumerate(ids) if id_[0] not in 'OPQ' and len(id_)==6]

    coef[c1] = np.array([3, 10,36,36,36,1,1,1,1,1])
    coef[c2] = np.array([23,10,26,36,36,1,1,1,1,1])

    for i in range(1,10):
        coef[:,-i-1] *= coef[:,-i]

    return np.sum(arr*coef,axis=-1)

msa1, lab1 = read_a3m(a3m1)
msa2, lab2 = read_a3m(a3m2)

# convert uniprot ids into integers
hash1 = uni2idx(lab1[1:])
hash2 = uni2idx(lab2[1:])

# find pairs of uniprot ids which are separated by at most 10
idx1,idx2 = np.where(np.abs(hash1[:,None]-hash2[None,:]) < 10)
print("pairs found: ", idx1.shape[0], idx2.shape[0])

# save paired alignment
with open('paired.a3m','wt') as f:
    # here we assume that the pair of first sequences from the two MSAs
    # is already paired (we know that these two proteins do interact)
    f.write('>query\n%s%s\n'%(msa1[0],msa2[0]))

    # save all other pairs
    for i,j in zip(idx1,idx2):
        f.write(">%s_%s\n%s%s\n"%(lab1[i+1],lab2[j+1],msa1[i+1],msa2[j+1]))
