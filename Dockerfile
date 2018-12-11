FROM nvidia/cuda:10.0-base-ubuntu16.04

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

# Install Miniconda and Python 3.6.5
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.5 \
 && conda clean -ya

# Install PyTorch with CUDA support
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install matplotlib, pandas, ffmpeg, and graphviz
RUN conda install -y matplotlib=2.2.3 pandas=0.23.4 ffmpeg=3.4 graphviz=2.40.1 \
 && conda clean -ya

# Use tkinter as the default matplotlib backend
RUN mkdir -p $HOME/.config/matplotlib \
 && echo "backend : TkAgg" > $HOME/.config/matplotlib/matplotlibrc

# Install other dependencies from pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Replace Pillow with the faster Pillow-SIMD (optional)
RUN pip uninstall -y pillow \
 && sudo apt-get update && sudo apt-get install -y gcc \
 && pip install pillow-simd==5.2.0.post0 \
 && sudo apt-get remove -y gcc \
 && sudo apt-get autoremove -y \
 && sudo rm -rf /var/lib/apt/lists/*

COPY --chown=user:user . /app
RUN pip install -U .

# Set the default command to python3
CMD ["python3"]
