import numpy as np

# 3LAA to tip atom conversion.
dict_3LAA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
                "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
                "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
                "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# 1LAA to number conversion.
aas = "ACDEFGHIKLMNPQRSTVWY-"
dict_1LAA_to_num = dict([(aas[a], a) for a in range(len(aas))])

# 3LAA to 1LAA conversion.
dict_3LAA_to_1LAA = {"ALA":"A", "CYS":"C", "ASP":"D", "ASN":"N", "GLU":"E",
                 "GLN":"Q", "PHE":"F", "HIS":"H", "ILE":"I", "GLY":"G",
                 "LEU":"L", "MET":"M", "ARG":"R", "LYS":"K", "PRO":"P",
                 "VAL":"V", "TYR":"Y", "TRP":"W", "SER":"S", "THR":"T"}