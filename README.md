# *RoseTTAFold* 
This package contains deep learning models and related scripts to run RoseTTAFold
This repository is the official implementation of RoseTTAFold: Accurate prediction of protein structures and interactions using a 3-track network.

## Installation

1. clone the package
```
git clone https://github.com/RosettaCommons/RoseTTAFold
cd RoseTTAFold
```

2. create conda environment using `RoseTTAFold-linux.yml` file and `folding-linux.yml` file. The latter required to run pyrosetta version only (run_pyrosetta_ver.sh).
```
conda env create -f RoseTTAFold-linux.yml
conda env create -f folding-linux.yml
```

3. download network weights (under Rosetta-DL Software license -- please see below)
While the code is licensed under the MIT License, the trained weights and data for RoseTTAFold are made available for non-commercial use only under the terms of the Rosetta-DL Software license. You can find details at https://files.ipd.uw.edu/pub/RoseTTAFold/Rosetta-DL_LICENSE.txt

```
wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz
tar xfz weights.tar.gz
```

4. download and install third-party software if you want to run the entire modeling script (run_pyrosetta_ver.sh)
```
./install_dependencies.sh
```

5. download sequence and structure databases
```
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates [10G]
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

Obtain a [PyRosetta licence](https://els2.comotion.uw.edu/product/pyrosetta) and install the package in the newly created `folding` conda environment ([link](http://www.pyrosetta.org/downloads)).


## Usage

```
cd example
../run_[pyrosetta, e2e]_ver.sh input.fa .
```

## Expected outputs
For the pyrosetta version, user will get five final models having estimated CA rms error at the B-factor column (model/model_[1-5].crderr.pdb).
For the end-to-end version, there will be a single PDB output having estimated residue-wise CA-lddt at the B-factor column (t000_.e2e.pdb).

## Links

* [Robetta server](https://robetta.bakerlab.org/) (RoseTTAFold option)

## Credit to performer-pytorch and SE(3)-Transformer codes
The code in the network/performer_pytorch.py is strongly based on [this repo](https://github.com/lucidrains/performer-pytorch) which is pytorch implementation of [Performer architecture](https://arxiv.org/abs/2009.14794).
The codes in network/equivariant_attention is from the original SE(3)-Transformer [repo](https://github.com/FabianFuchsML/se3-transformer-public) which accompanies [the paper](https://arxiv.org/abs/2006.10503) 'SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks' by Fabian et al.


## References

M Baek, et al., Accurate prediction of protein structures and interactions using a 3-track network, bioRxiv (2021). [link](https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1)

