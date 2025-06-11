# ------------------------------------------------------------
# Base image: Ubuntu 20.04 + CUDA 11.1 + cuDNN8
# ------------------------------------------------------------
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# 非互動式安裝、設定時區為 UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
 && echo "UTC" > /etc/timezone

# 安裝系統工具（包含 tmux）
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential wget curl git ca-certificates \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
      libffi-dev libssl-dev cmake swig tmux \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 安裝 Miniconda（安放至 /root/miniconda）
# ------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O /tmp/miniconda.sh \
 && /bin/bash /tmp/miniconda.sh -b -p /root/miniconda \
 && rm /tmp/miniconda.sh \
 && /root/miniconda/bin/conda clean -a -y \
 && ln -s /root/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# 讓 conda 指令可用
ENV PATH=/root/miniconda/bin:$PATH

# 安裝 mamba（更快的 conda 取代方案）
RUN conda install -n base -c conda-forge mamba -y

# 使用 bash -lc 使下面的 source activate 生效
SHELL ["/bin/bash", "-lc"]

# ------------------------------------------------------------
# 建立 py36 環境並安裝所有 Python 套件
# ------------------------------------------------------------
RUN mamba create -n py36 python=3.6.9 -y \
 && mamba install -n py36 -c conda-forge \
      numpy scipy pandas matplotlib seaborn -y \
 && mamba install -n py36 -c pytorch -c nvidia \
      pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -y \
 && source activate py36 \
 && pip install \
      tensorflow==2.1.0 \
      tensorboard \
      gym \
      stable-baselines3 \
      'ray[rllib]' \
      tianshou \
      eclipse-sumo==1.11.0 \
      traci \
      sumolib \
      torch \
 && conda clean -afy

# 每次進 bash 都自動啟用 py36
RUN echo "conda activate py36" >> /root/.bashrc

# ------------------------------------------------------------
# 預設使用 py36 環境、並設定 SUMO_HOME
# ------------------------------------------------------------
ENV PATH=/root/miniconda/envs/py36/bin:/root/miniconda/bin:$PATH \
    SUMO_HOME=/root/miniconda/envs/py36/share/sumo

# 工作目錄
WORKDIR /workspace

# 預設進入 bash，且已在 py36 環境中
CMD ["bash"]
