import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import time
from .resnet import *
from .model import *
from .deepLearningUtils import *

# Loads in files for one prediction
def getData(tmp, outfolder, distogram):
    data = np.load(tmp)
        
    # 3D information
    idx = data["idx"]
    val = data["val"]
    
    # 1D information
    angles = np.stack([np.sin(data["phi"]), np.cos(data["phi"]), np.sin(data["psi"]), np.cos(data["psi"])], axis=-1)
    obt = data["obt"].T
    prop = data["prop"].T
    
    # 2D information
    orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
    orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
    maps = data["maps"]
    tbt = data["tbt"].T
    sep = seqsep(tbt.shape[0])
    
    # Transformation
    tbt[:,:,0] = transfomer(tbt[:,:,0])
    maps = transfomer(maps)

    return (idx, val),\
            np.concatenate([angles, obt, prop], axis=-1),\
            np.concatenate([tbt, maps, orientations, sep, distogram], axis=-1)
    
# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def transfomer(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def getDistribution(outfolder):
    path = outfolder
    tbts = [np.load(path+"/"+f)["tbt"][0,:,:] for f in os.listdir(path) if isfile(join(path,f)) and ".features.npz" in f]
    for i in range(len(tbts)-1):
        if not tbts[i].shape == tbts[i+1].shape:
            print("All pdbs in the input folder need to have the same size.")
    tbt = np.array(tbts)
    transformed = transfomer(tbt, cutoff=6, scaling=1.0)
    digitization = np.arange(0.25,5.1,0.25)
    binned = np.eye(len(digitization)+1)[np.digitize(transformed, digitization)]
    normalized = np.sum(binned, axis=0)/tbt.shape[0]
    np.save(join(outfolder, "dist.npy"), normalized)
    
def predict(samples, distogram, modelpath, outfolder, verbose=False, transpose=False):
    n_models = 2 
    for i in range(1, n_models):
        modelname = modelpath+"_rep"+str(i)
        if verbose: print("Loading", modelname)
        
        model = Model(obt_size=70,
                      tbt_size=58,
                      prot_size=None,
                      num_chunks=5,
                      optimizer="adam",
                      mask_weight=0.33,
                      lddt_weight=10.0,
                      name=modelname,
                      verbose=False)
        model.load()
            
        for j in range(len(samples)):
            if verbose: print("Predicting for", samples[j], "(network rep"+str(i)+")") 
            tmp = join(outfolder, samples[j]+".features.npz")
            batch = getData(tmp, outfolder, distogram)
            lddt, estogram, mask = model.predict2(batch)
            if transpose:
                estogram = (estogram + np.transpose(estogram, [1,0,2]))/2
                mask = (mask + mask.T)/2
            np.savez_compressed(join(outfolder, samples[j]+".npz"),
                                lddt = lddt,
                                estogram = estogram,
                                mask = mask)
                
def merge(samples, outfolder, verbose=False):
    for j in range(len(samples)):
        if verbose: print("Merging", samples[j])

        lddt = []
        estogram = []
        mask = []
        for i in range(1,5):
            temp = np.load(join(outfolder, samples[j]+".rep"+str(i)+".npz"))
            lddt.append(temp["lddt"])
            estogram.append(temp["estogram"])
            mask.append(temp["mask"])

        # Averaging
        lddt = np.mean(lddt, axis=0)
        estogram = np.mean(estogram, axis=0)
        mask = np.mean(mask, axis=0)

        # Saving
        np.savez_compressed(join(outfolder, samples[j]+".npz"),
                lddt = lddt,
                estogram = estogram,
                mask = mask)
                
def clean(samples, outfolder, verbose=False):
    for i in range(len(samples)):
        if verbose: print("Removing", join(outfolder, samples[i]+".features.npz"))
        os.remove(join(outfolder, samples[i]+".features.npz"))
