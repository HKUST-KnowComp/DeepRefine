FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SWANLAB_API_KEY="**"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl wget ca-certificates build-essential \
    ninja-build cmake python3.10-dev procps iproute2 libssl-dev libnuma1 libnuma-dev \
    emacs vim tmux htop less jq tree unzip screen \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# copy REAfiner project into the image
COPY . /workspace

RUN python3 -m pip install --upgrade pip

# ===== Core deep learning environment (align with your current atlastune conda environment) =====
# Python 3.10.x is installed above, here we first install the GPU version of PyTorch, then install the other dependencies according to atlastune_environment.yml

RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

COPY docker/atlastune_environment.yml /workspace/docker/atlastune_environment.yml

RUN sed -n '/- pip:/,$p' /workspace/docker/atlastune_environment.yml \
    | sed '1d' \
    | grep '^\s*-\s' \
    | sed 's/^\s*-\s*//' \
    | grep -v '^torch==' \
    | grep -v '^torchvision==' \
    | grep -v '^torchaudio==' \
    | grep -v '^flash-attn==' \
    > /tmp/requirements_atlastune_full.txt \
    && python3 -m pip install -r /tmp/requirements_atlastune_full.txt

# 4. Finally install flash-attn (at this point the packaging dependencies are in place)
RUN python3 -m pip install --no-build-isolation flash-attn==2.8.3

RUN python3 -m pip install jupyterlab

# Make python able to directly import the code in this repository
ENV PYTHONPATH=/workspace/code:${PYTHONPATH:-}

# The script requires running in the root of the project, set the default working directory to code
WORKDIR /workspace/code

RUN chmod +x /workspace/docker/*.sh 2>/dev/null || true

ENTRYPOINT ["bash"]

