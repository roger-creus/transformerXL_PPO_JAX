# Base image with CUDA 12.2, cuDNN, Ubuntu 22.04
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Configure Conda channels
RUN conda config --add channels conda-forge && \
    conda config --add channels defaults && \
    conda tos accept --override-channels --channel conda-forge --channel defaults

# Copy environment YAML
COPY environment.yml /tmp/environment.yml

# Create Conda environment
RUN conda update -n base -c defaults conda -y && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Create entrypoint script to activate the environment
RUN echo '#!/bin/bash' > /opt/entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /opt/entrypoint.sh && \
    echo 'conda activate trxl' >> /opt/entrypoint.sh && \
    echo 'exec "$@"' >> /opt/entrypoint.sh && \
    chmod +x /opt/entrypoint.sh

# Set correct entrypoint
ENTRYPOINT ["/opt/entrypoint.sh"]

# Default command
CMD ["python"]
