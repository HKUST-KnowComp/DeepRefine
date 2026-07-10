FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SWANLAB_API_KEY="**"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl wget ca-certificates build-essential \
    ninja-build cmake python3.10-dev procps iproute2 libssl-dev libnuma1 libnuma-dev \
    emacs vim tmux htop less jq tree unzip screen \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace

# Copy the entire DeepRefine project (code + data + configs + scripts)
COPY . /workspace

RUN python3 -m pip install --upgrade pip

# ===== Core deep learning environment (align with atlastune conda environment) =====
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Install frozen atlastune deps as-is (--no-deps): this is a working conda
# pip-freeze snapshot; strict resolution conflicts (e.g. numpy 1.26 vs opencv 4.12).
# Skip torch stack (installed above) and nvidia CUDA wheels already pulled by torch.
RUN grep -vE '^(torch|torchvision|torchaudio|flash-attn|nvidia-cublas|nvidia-cuda|nvidia-cudnn|nvidia-cufft|nvidia-cufile|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-cusparselt|nvidia-nccl|nvidia-nvjitlink|nvidia-nvtx)==' \
      /workspace/docker/requirements-atlastune.txt \
      > /tmp/requirements_atlastune_full.txt \
    && python3 -m pip install --no-deps -r /tmp/requirements_atlastune_full.txt

# Install flash-attn after packaging deps are in place
RUN python3 -m pip install --no-build-isolation flash-attn==2.8.3

RUN python3 -m pip install jupyterlab \
    && python3 -m pip install -e /workspace

# Import paths for this repo (verl, autorefiner, autograph, ...)
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN chmod +x /workspace/docker/*.sh /workspace/scripts/**/*.sh 2>/dev/null || true

ENTRYPOINT ["bash"]
