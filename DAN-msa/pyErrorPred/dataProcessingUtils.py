from pyrosetta import *
import math
import numpy as np
import pandas as pd
import csv
import pkg_resources

####################
# INDEXERS/MAPPERS 
####################
# Assigning numbers to 3 letter amino acids.
residues= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(residues[i], i) for i in range(len(residues))])

# Mapping 3 letter AA to 1 letter AA (e.g. ALA to A)
oneletter = ["A", "R", "N", "D", "C", \
 "Q", "E", "G", "H", "I", \
 "L", "K", "M", "F", "P", \
 "S", "T", "W", "Y", "V"]
aanamemap = dict([(residues[i], oneletter[i]) for i in range(len(residues))])

#################
# BLOSUM SCORES 
#################
# Dictionary for Blosum score.
# Keys are 1 letter residues and it returns corresponding slice of blosum
location = pkg_resources.resource_filename(__name__, 'data/blosum62.txt')
blosum = [i.strip().split() for i in open(location).readlines()[1:-1]]
blosummap = dict([(l[0], np.array([int(i) for i in l[1:]])/10.0) for l in blosum])

####################
# ROSETTA ENERGIES 
####################
energy_terms = [pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,\
                pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_wtd,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,\
                pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,\
                pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc]
energy_names = ["fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec", "hbond_bb_sc", "hbond_sc"]

###################
# MEILER FEATIRES
###################
location = pkg_resources.resource_filename(__name__, "data/labeled_features_meiler2001.csv")
temp = pd.read_csv(location).values
meiler_features = dict([(t[0], t[1:]) for t in temp])

###################
# ATYPE CHANNELS
###################
atypes = {}
types = {}
ntypes = 0
script_dir = os.path.dirname(__file__)
location = pkg_resources.resource_filename(__name__, "data/groups20.txt")
with open(location, 'r') as f:
    data = csv.reader(f, delimiter=' ')
    for line in data:
        if line[1] in types:
            atypes[line[0]] = types[line[1]]
        else:
            types[line[1]] = ntypes
            atypes[line[0]] = ntypes
            ntypes += 1