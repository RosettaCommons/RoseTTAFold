import numpy as np
import pyrosetta
from scipy.spatial import distance, distance_matrix
from .conversion import *

# Gets distance for various atoms given a pose
def get_distmap_deprecated(pose, atom1="CA", atom2="CA", default="CA"):
    out = np.zeros((pose.size(), pose.size()))
    for i in range(1, pose.size()+1):
        for j in range(1, pose.size()+1):
            r = pose.residue(i)
            if type(atom1) == str:
                if r.has(atom1):
                    p1 = np.array(r.xyz(atom1))
                else:
                    p1 = np.array(r.xyz(default))
            else:
                p1 = np.array(r.xyz(atom1.get(r.name(), default)))
                
            r = pose.residue(j)
            if type(atom2) == str:
                if r.has(atom2):
                    p2 = np.array(r.xyz(atom2))
                else:
                    p2 = np.array(r.xyz(default))
            else:
                p2 = np.array(r.xyz(atom2.get(r.name(), default)))
                
            dist = distance.euclidean(p1,p2)
            out[i-1, j-1] = dist
    return out

# Gets distance for various atoms given a pose
def get_distmaps(pose, atom1="CA", atom2="CA", default="CA"):
    psize = pose.size()
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(1, psize+1):
        r = pose.residue(i)
        
        if type(atom1) == str:
            if r.has(atom1):
                xyz1[i-1, :] = np.array(r.xyz(atom1))
            else:
                xyz1[i-1, :] = np.array(r.xyz(default))
        else:
            xyz1[i-1, :] = np.array(r.xyz(atom1.get(r.name(), default)))
    
        if type(atom2) == str:
            if r.has(atom2):
                xyz2[i-1, :] = np.array(r.xyz(atom2))
            else:
                xyz2[i-1, :] = np.array(r.xyz(default))
        else:
            xyz2[i-1, :]  = np.array(r.xyz(atom2.get(r.name(), default)))

    return distance_matrix(xyz1, xyz2)

# Gets torsions given a pose
def getTorsions(pose):
    p = pose
    torsions=np.zeros((p.size(),3))
    for i in range(p.total_residue()):
        torsions[(i,0)]=p.phi(i+1)
        torsions[(i,1)]=p.psi(i+1)
        torsions[(i,2)]=p.omega(i+1)
    return torsions

# Gets sequence given a pose
def get_sequence(pose):
    p = pose
    seq=[p.residue(i).name() for i in range(1,p.size()+1)]
    return seq

# Get euler angles of pairs of residues
def getEulerOrientation(pose):
    # Getting Euler angle of res-res oriention
    trans_z=np.zeros((pose.size(),pose.size(),3))
    rot_z=np.zeros((pose.size(),pose.size(),3))
    for i in range(1,pose.size()+1):
        for j in range(1,pose.size()+1):
            if i == j: continue
            rt6=pyrosetta.rosetta.core.scoring.motif.get_residue_pair_rt6(pose,i,pose,j)
            trans_z[i-1][j-1] = np.array([rt6[1],rt6[2],rt6[3]])
            rot_z[i-1][j-1] = np.array([rt6[4],rt6[5],rt6[6]])
    
    # Conversion to radian space
    trans_z = np.deg2rad(trans_z)
    rot_z = np.deg2rad(rot_z)
    
    # Getting sin and consine of respective angles
    output = np.concatenate([trans_z, rot_z], axis=2)
    return output

# Given a pose and scorefunction, returns one body and two body terms of totalE.
def getEnergy(p, scorefxn):
    nres=p.size()
    res_pair_energy_z=np.zeros( (nres,nres) )
    res_energy_no_two_body_z=np.zeros ( (nres) )

    totE=scorefxn(p)
    energy_graph=p.energies().energy_graph()
    twobody_terms = p.energies().energy_graph().active_2b_score_types()
    onebody_weights = pyrosetta.rosetta.core.scoring.EMapVector()
    onebody_weights.assign(scorefxn.weights())

    for term in twobody_terms:
        if 'intra' not in pyrosetta.rosetta.core.scoring.name_from_score_type(term):
            onebody_weights.set(term, 0)
            
    for i in range(1,nres+1):
        res_energy_no_two_body_z[i-1] = p.energies().residue_total_energies(i).dot(onebody_weights)
            
        for j in range(1,nres+1):
            if i == j: continue
            edge=energy_graph.find_edge(i,j)
            if edge is None:
                energy=0.
            else:
                res_pair_energy_z[i-1][j-1]= edge.fill_energy_map().dot(scorefxn.weights())
                
    return res_energy_no_two_body_z, res_pair_energy_z

# Get 1 hot encoded aminoacides
def get1hotAA(pose, indecies=dict_1LAA_to_num):
    AAs = [i.split(":")[0] for i in get_sequence(pose)]
    output = np.zeros((pose.size(), len(dict_1LAA_to_num)))
    for i in range(len(AAs)):
        output[i, indecies[dict_3LAA_to_1LAA[AAs[i]]]] = 1
    return output
