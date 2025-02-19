# Use the NVIDIA CUDA devel image as base
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Install required dependencies for Miniconda installation
RUN apt-get update && apt-get install -y wget bzip2

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# (Optional) Update conda and clean up
RUN conda update -n base -c defaults conda -y && \
    conda clean --all -y

# Create a new conda environment 'gpu_env' with required packages
# Install mamba in the base image
RUN conda install -n base -c conda-forge mamba -y

# Create a new environment and install dependencies using mamba
RUN mamba create -n gpu_env python=3.9 -y && \
    mamba install -n gpu_env -c rapidsai -c nvidia -c conda-forge rapids=23.12 cudf cuml cupy scikit-learn pandas matplotlib seaborn -y

    CMD ["/bin/bash"]


