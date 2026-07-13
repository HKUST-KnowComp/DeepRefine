FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl wget ca-certificates build-essential \
    ninja-build cmake python3.10-dev procps iproute2 libssl-dev libnuma1 libnuma-dev \
    emacs vim tmux htop less jq tree unzip screen \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Put project under /opt/DeepRefine — Runpod mounts an empty volume on /workspace
# and would hide anything copied there.
WORKDIR /opt/DeepRefine
COPY . /opt/DeepRefine

RUN python3 -m pip install --upgrade pip

# ===== Core deep learning environment (align with atlastune conda environment) =====
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Install frozen atlastune deps as-is (--no-deps): this is a working conda
# pip-freeze snapshot; strict resolution conflicts (e.g. numpy 1.26 vs opencv 4.12).
# Skip torch stack (installed above) and nvidia CUDA wheels already pulled by torch.
RUN grep -vE '^(torch|torchvision|torchaudio|flash-attn|nvidia-cublas|nvidia-cuda|nvidia-cudnn|nvidia-cufft|nvidia-cufile|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-cusparselt|nvidia-nccl|nvidia-nvjitlink|nvidia-nvtx)==' \
      /opt/DeepRefine/docker/requirements-atlastune.txt \
      > /tmp/requirements_atlastune_full.txt \
    && python3 -m pip install --no-deps -r /tmp/requirements_atlastune_full.txt

# Install flash-attn after packaging deps are in place
RUN python3 -m pip install --no-build-isolation flash-attn==2.8.3

RUN python3 -m pip install jupyterlab \
    && python3 -m pip install -e /opt/DeepRefine

# Import paths for this repo (verl, autorefiner, autograph, ...)
ENV PYTHONPATH=/opt/DeepRefine
WORKDIR /opt/DeepRefine

RUN chmod +x /opt/DeepRefine/docker/*.sh /opt/DeepRefine/scripts/**/*.sh 2>/dev/null || true

# Install and configure sshd late so earlier heavy layers stay cached.
# Required for Runpod direct TCP SSH (Cursor/VS Code Remote-SSH).
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server     && mkdir -p /var/run/sshd /root/.ssh     && chmod 700 /root/.ssh     && ssh-keygen -A     && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config     && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config     && sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config     && echo 'Port 22' >> /etc/ssh/sshd_config     && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
