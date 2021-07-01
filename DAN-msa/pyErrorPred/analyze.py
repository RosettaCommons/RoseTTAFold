import numpy as np
from scipy.spatial import distance, distance_matrix

# Given a pdb file,returns binary map for inter monomer interactions.
def get_interaction_map(filename):
    f = open(filename)
    lines = f.readlines()
    chain_names = np.array([l[21] for l in lines if "ATOM" in l and l[12:16]==" CA "])
    f.close()
    codes = ["A","B"]
    output = np.zeros((len(chain_names), len(chain_names)))
    output2 = []
    for c in codes:
        temp = np.zeros((np.sum(chain_names==c), len(chain_names)))
        temp[:, chain_names==c] = 1
        output[chain_names==c, :] = temp
        
        temp2 = np.zeros((len(chain_names), len(chain_names)))
        temp2[chain_names==c, :] = temp
        output2.append(temp2)
    return output*-1+1, output2

# Returns a cb contactmap given pdb.
def get_contact_map(filename, threshold=10):
    f = open(filename)
    lines = f.readlines()
    xyz = np.array([np.array(l[31:55].split()).astype(float) for l in lines if "ATOM" in l and l[12:16]==" CA "])
    distmap = distance_matrix(xyz, xyz)
    distmap = distmap<threshold
    return distmap

def get_lddt(estogram, mask, center=7, weights=[1,1,1,1]):  
    # Remove diagonal from the mask.
    mask = np.multiply(mask, np.ones(mask.shape)-np.eye(mask.shape[0]))
    # Masking the estogram except for the last cahnnel
    masked = np.transpose(np.multiply(np.transpose(estogram, [2,0,1]), mask), [1,2,0])

    p0 = np.sum(masked[:,:,center], axis=-1)
    p1 = np.sum(masked[:,:,center-1]+masked[:,:,center+1], axis=-1)
    p2 = np.sum(masked[:,:,center-2]+masked[:,:,center+2], axis=-1)
    p3 = np.sum(masked[:,:,center-3]+masked[:,:,center+3], axis=-1)
    p4 = np.sum(mask, axis=-1)

    # Only work on parts where interaction happen
    output = np.divide((weights[0]*p0 + weights[1]*(p0+p1) + weights[2]*(p0+p1+p2) + weights[3]*(p0+p1+p2+p3))/np.sum(weights), p4, where=p4!=0)
    return output[p4!=0]