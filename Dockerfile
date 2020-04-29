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

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# Setup conda environment
COPY environment.yml .
RUN conda env update --name base

# Use tkinter as the default matplotlib backend
RUN mkdir -p $HOME/.config/matplotlib \
 && echo "backend : TkAgg" > $HOME/.config/matplotlib/matplotlibrc

# Install other dependencies from pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Replace Pillow with the faster Pillow-SIMD (optional)
RUN pip uninstall -y pillow \
 && sudo apt-get update && sudo apt-get install -y gcc libjpeg8-dev zlib1g-dev \
 && pip install pillow-simd==6.2.2post1 \
 && sudo apt-get remove -y gcc \
 && sudo apt-get autoremove -y \
 && sudo rm -rf /var/lib/apt/lists/*

COPY --chown=user:user . /app
RUN pip install -U .

# Set the default command to python3
CMD ["python3"]
