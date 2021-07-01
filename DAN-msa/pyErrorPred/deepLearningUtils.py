import os
import tensorflow as tf
import numpy as np
import sys
import glob
from os import listdir
from os.path import isfile, join

class dataloader:
    
    def __init__(self,
                 proteins, # list of proteins to load
                 datadir="/projects/casp/dldata/", # Base directory for all protein data
                 lengthmax=280, # Limit to the length of proteins, if bigger we ignore.
                 load_dtype=np.float32, # Data type to load with
                 digitization1 = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0],
                 digitization2 = np.arange(-20.5,20.6,1.0),
                 features=["obt", "prop", "dist"], # Kinds of features to incooperate.
                 verbose=False,
                 distribution=False,
                 include_native=True,
                 include_native_dist=True,
                 distance_cutoff = 6,
                 trrosetta_base = "/projects/ml/DLdecoys/trRefine/"
                ):
        
        self.n = {}
        self.base_dist = {}
        self.samples_dict = {}
        self.sizes = {}
        
        self.load_dtype = load_dtype
        self.digitization1 = digitization1
        self.digitization2 = digitization2
        self.datadir = datadir
        self.features = features
        self.verbose = verbose
        self.distribution = distribution
        self.include_native = include_native
        self.include_native_dist = include_native_dist
        self.distance_cutoff = distance_cutoff
        self.trrosetta_base = trrosetta_base
        if self.verbose: print("features:", self.features)
            
        # Loading file availability
        temp = []
        #np.random.seed(7)
        for p in proteins:
            # check datadir
            if not os.path.exists(self.datadir + p):
                #print ("Exclude, no features: ", p)
                continue
            # check predicted distogram
            if not os.path.exists(self.trrosetta_base + p):
                #print ("Exclude, no distogram: ", p)
                continue
            #
            path = datadir+p+"/"
            if not os.path.exists(path+"native.features.npz"):
                continue
            samples = glob.glob(path + "*.features.npz")

            # Removing native from distribution if you are using distribution option
            if not self.include_native:
                samples = [s for s in samples if not "native" in s]
            #
            sub_s = glob.glob(self.trrosetta_base + "%s/sub_*"%p)
            sub_s.sort()
            base_dist = "%s/naive.npz"%(sub_s[-1])
            #
            np.random.shuffle(samples)
                    
            if len(samples) > 0:
                print (samples[0])
                length = np.load(samples[0])["tbt"].shape[-1]
                if length < lengthmax:
                    temp.append(p)
                    self.base_dist[p] = base_dist
                    self.samples_dict[p] = samples
                    self.n[p] = len(samples)
                    self.sizes[p] = length

        # Make a list of proteins
        self.proteins = temp
        self.index = np.arange(len(self.proteins))
        np.random.shuffle(self.index)
        self.cur_index = 0

    def next(self, transform=True, pindex=-1, tr_idx=None):
        pname = self.proteins[self.index[self.cur_index]]
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.n[pname]))
        sample = self.samples_dict[pname][pindex]
        psize = self.sizes[pname]
        data = np.load(sample)
        #
        if self.trrosetta_base != None:
            if tr_idx != None: # testing
                sub_s = glob.glob(self.trrosetta_base + "%s/new_0/naive_rolled.npz"%pname)
                sub_s.extend(glob.glob(self.trrosetta_base + "%s/sub_0/naive.npz"%pname))
                base_dist = sub_s[0]
            else: # training
                sub_s = glob.glob(self.trrosetta_base + "%s/sub_[0-2]/naive.npz"%pname)
                sub_s.extend(glob.glob(self.trrosetta_base + "%s/sub_[0-2]/templ.npz"%pname))
                sub_s.extend(glob.glob(self.trrosetta_base + "%s/new_[0-2]/naive_rolled.npz"%pname))
                sub_s.extend(glob.glob(self.trrosetta_base + "%s/new_[0-2]/templ_rolled.npz"%pname))
                base_dist = np.random.choice(sub_s)
            distogram_pred = np.load(base_dist)['dist']
        
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
        sep = seqsep(psize)
        
        if self.distribution:
            if self.include_native_dist:
                dist = np.load(self.datadir+pname+"/dist.npy")
            else:
                dist = np.load(self.datadir+pname+"/dist2.npy")
        
        # Get target
        native = np.load(self.datadir+pname+"/native.features.npz")["tbt"][0]
        estogram = get_estogram((tbt[:,:,0], native), self.digitization1)
        
        # Transform input distance
        if transform:
            tbt[:,:,0] = f(tbt[:,:,0])
            maps = f(maps, cutoff=self.distance_cutoff)
        
        self.cur_index += 1
        if self.cur_index == len(self.proteins):        
            self.cur_index = 0 
            np.random.shuffle(self.index)
            
            
        if self.verbose:
            print(angles.shape, obt.shape, prop.shape)
            print(tbt.shape, maps.shape, euler.shape, orientations.shape, sep.shape)
            
        if self.distribution:
            return (idx, val),\
                    np.concatenate([angles, obt, prop], axis=-1),\
                    np.concatenate([maps, orientations, sep, distogram_pred, dist], axis=-1),\
                    (estogram, native)
                    #np.concatenate([tbt, maps, euler, orientations, sep, distogram_pred, dist], axis=-1),\
        else:
            if self.trrosetta_base != None:
                return (idx, val),\
                        np.concatenate([angles, obt, prop], axis=-1).astype(np.float32),\
                        np.concatenate([tbt, maps, orientations, sep, distogram_pred], axis=-1).astype(np.float32),\
                        (estogram, native)
                        #np.concatenate([tbt, maps, euler, orientations, sep, distogram_pred], axis=-1).astype(np.float32),\
            else:
                return (idx, val),\
                        np.concatenate([angles, obt, prop], axis=-1).astype(np.float32),\
                        np.concatenate([tbt, maps, euler, orientations, sep], axis=-1).astype(np.float32),\
                        (estogram, native)
    
        
def f(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def get_estogram(XY, digitization):
    (X,Y) = XY
    residual = X-Y
    estogram = np.eye(len(digitization)+1)[np.digitize(residual, digitization)]
    return estogram

# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def apply_label_smoothing(x, alpha=0.2, axis=-1):
    minind = 0
    maxind = x.shape[axis]-1
    
    # Index of true and semi-true labels
    index = np.argmax(x, axis=axis)
    lower = np.clip(index-1, minind, maxind)
    higher = np.clip(index+1, minind, maxind)
    
    # Location-aware label smoothing
    true = np.eye(maxind+1)[index]*(1-alpha)
    semi_lower = np.eye(maxind+1)[lower]*(alpha/2)
    semi_higher= np.eye(maxind+1)[higher]*(alpha/2)
    
    return true+semi_lower+semi_higher

# Getting masks
def getMask(exclude):
    feature2D = [("distance",1), ("rosetta",9), ("distance",4), ("orientation",18), ("seqsep",1)]
    feature1D = [("angles",10), ("rosetta",4), ("ss",4), ("aa", 52)]
    for e in exclude:
        if e not in [i[0] for i in feature2D] and e not in [i[0] for i in feature1D]:
            print("Feature names do not exist.")
            print([i[0] for i in feature1D])
            print([i[0] for i in feature2D])
            return -1
    mask = []
    temp = []
    index = 0
    for f in feature1D:
        for i in range(f[1]):
            if f[0] in exclude: temp.append(index)
            index+=1
    mask.append(temp)
    temp = []
    index = 0
    for f in feature2D:
        for i in range(f[1]):
            if f[0] in exclude: temp.append(index)
            index+=1
    mask.append(temp)
    return mask
