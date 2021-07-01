# *RoseTTAFold* 
This package contains deep learning models and related scripts to run RoseTTAFold

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
tar xf weights.tar.gz
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
tar xf UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xf bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates [8.3G]
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xf pdb100_2021Mar03.tar.gz
```

Obtain a [PyRosetta licence](https://els2.comotion.uw.edu/product/pyrosetta) and install the package in the newly created `folding` conda environment ([link](http://www.pyrosetta.org/downloads)).


## Usage

```
mkdir -p example/
./run_pipeline.sh example/input.fa example/
```

## Links

* [Robetta server](https://robetta.bakerlab.org/) (RoseTTAFold option)


## References

M Baek, et al., Accurate prediction of protein structures and interactions using a 3-track network, bioRxiv (2021). [link](https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1)

