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

# Install mamba in the base image
RUN conda install -n base -c conda-forge mamba -y

# Create a new environment and install dependencies using mamba
RUN mamba create -n gpu_env python=3.10 -y && \
    mamba install -n gpu_env -c rapidsai -c nvidia -c conda-forge cudf cuml cupy scikit-learn pandas matplotlib seaborn -y

# Add auto-activation of gpu_env to .bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate gpu_env" >> /root/.bashrc

# Copy the Python scripts into the container
COPY "AI_folder(RF)/GPU_RUNNER_DOCKER.py" /app/GPU_RUNNER_DOCKER.py
COPY "AI_folder(RF)/RF_MODEL_GPU.py" /app/RF_MODEL_GPU.py

# Create a directory for the CSV file and copy it into the container.
RUN mkdir -p /app/data
COPY ["LabeledData/Old label/Merged Data/Random_Merged_Data_12-06-30.csv", "/app/data/Random_Merged_Data_12-06-30.csv"]
WORKDIR /app

# Start container with interactive bash shell (which now auto-activates gpu_env)
CMD ["bash", "-c", "exec bash"]
