FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
ARG ROSETTACOMMONS_CONDA_USERNAME
ARG ROSETTACOMMONS_CONDA_PASSWORD

RUN apt-get update

RUN apt-get install -y wget libgomp1 unzip && rm -rf /var/lib/apt/lists/*

RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /var/conda\
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH /var/conda/bin:$PATH

RUN conda --version

COPY . /RoseTTaFold
WORKDIR /RoseTTaFold

RUN conda config --set remote_max_retries 5
RUN conda config --set remote_backoff_factor 20

RUN conda env create -q -f RoseTTAFold-linux.yml
RUN conda env create -q -f folding-linux.yml

RUN conda config --add channels https://${ROSETTACOMMONS_CONDA_USERNAME}:${ROSETTACOMMONS_CONDA_PASSWORD}@conda.graylab.jhu.edu
#installing pyrosetta into a base image so it gets cached between builds
RUN conda install -n folding pyrosetta=2020.45

RUN chmod +x install_dependencies.sh
RUN ./install_dependencies.sh

ENV PATH /RoseTTaFold:$PATH

WORKDIR /tmp
