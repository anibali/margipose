FROM nvidia/cuda:9.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.4 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Ensure conda version is at least 4.4.11
# (because of this issue: https://github.com/conda/conda/issues/6811)
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya

# Install GPU computing libraries
RUN conda install -y -c pytorch \
    cuda91=1.0 \
    magma-cuda91=2.3.0 \
    cudnn=7.0.5 \
 && conda clean -ya

# Install PyTorch
RUN conda install -y -c pytorch \
    pytorch=0.3.1 \
    torchvision=0.2.0 \
 && conda clean -ya

# Install OpenCV, FFmpeg, and Graphviz
RUN conda install --no-update-deps -y -c conda-forge opencv=3.3.0 ffmpeg=3.2.4 graphviz=2.38.0 \
 && conda clean -ya
RUN sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx \
 && sudo rm -rf /var/lib/apt/lists/*

# Use tkinter as the default matplotlib backend
RUN mkdir -p $HOME/.config/matplotlib \
 && echo "backend : TkAgg" > $HOME/.config/matplotlib/matplotlibrc

# Install other dependencies from pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Replace Pillow with the faster Pillow-SIMD (optional)
RUN pip uninstall -y pillow \
 && sudo apt-get update && sudo apt-get install -y gcc \
 && pip install pillow-simd==4.3.0.post0 \
 && sudo apt-get remove -y gcc \
 && sudo apt-get autoremove -y \
 && sudo rm -rf /var/lib/apt/lists/*

COPY --chown=user:user . /app
RUN pip install -e .

# Set the default command to python3
CMD ["python3"]
