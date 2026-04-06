#!/bin/bash

TMP_INSTALL_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_INSTALL_DIR"' EXIT

if [ -x "/usr/bin/gcc-12" ]; then
    echo "GCC already installed"
else
    sudo apt-get install -y gcc-12
fi

if [ -x "/usr/bin/make" ]; then
    echo "Make already installed"
else
    sudo apt-get install -y make
fi

CONDA_BASE=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/conda"
fi

if [ -n "$CONDA_BASE" ]; then
    echo "conda is already installed at $CONDA_BASE"
else
    MINICONDA="${MINICONDA:-Miniconda3-latest-Linux-x86_64.sh}"

    curl -o "$TMP_INSTALL_DIR/$MINICONDA" "https://repo.anaconda.com/miniconda/${MINICONDA}"

    EXPECTED_SHA256=$(curl -s https://repo.anaconda.com/miniconda/ \
        | grep -A4 "${MINICONDA}" \
        | grep -oE '[a-f0-9]{64}' \
        | head -1)

    if [ -z "$EXPECTED_SHA256" ]; then
        echo "Failed to fetch SHA256 hash from repo.anaconda.com"
        echo "Are you sure your miniconda installation exists?"
        exit 1
    fi

    ACTUAL_SHA256=$(sha256sum "$TMP_INSTALL_DIR/$MINICONDA" | awk '{print $1}')

    if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
        echo "SHA256 mismatch:"
        echo "  Expected: $EXPECTED_SHA256"
        echo "  Got:      $ACTUAL_SHA256"
        echo "Please check where the script is being fetched from"
        exit 1
    fi
    echo "Miniconda installer verified with hash: $ACTUAL_SHA256"

    bash "$TMP_INSTALL_DIR/$MINICONDA" -b -p "$HOME/miniconda3" || { echo "Miniconda installation failed"; exit 1; }
    CONDA_BASE="$HOME/miniconda3"
    
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

CUDA_VER="${CUDA_VER:-12.8.0}"
CUDA_MAJOR_MINOR="${CUDA_VER%.*}"

if [ -x "/usr/local/cuda-${CUDA_MAJOR_MINOR}/bin/nvcc" ]; then
    echo "CUDA ${CUDA_VER} already installed"
else
    CUDA="${CUDA:-cuda_12.8.0_570.86.10_linux.run}"
    wget -P "$TMP_INSTALL_DIR" "https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${CUDA}"

    EXPECTED_MD5=$(curl -s "https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/docs/sidebar/md5sum.txt" \
        | grep "${CUDA}" \
        | grep -oE '[a-f0-9]{32}' \
        | head -1)

    if [ -z "$EXPECTED_MD5" ]; then
        echo "Failed to fetch MD5 hash from NVIDIA"
        exit 1
    fi

    ACTUAL_MD5=$(md5sum "$TMP_INSTALL_DIR/$CUDA" | awk '{print $1}')

    if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
        echo "MD5 mismatch:"
        echo "  Expected: $EXPECTED_MD5"
        echo "  Got:      $ACTUAL_MD5"
        exit 1
    fi
    echo "CUDA installer verified with hash: $ACTUAL_MD5"

    sudo bash "$TMP_INSTALL_DIR/$CUDA" --silent --driver --toolkit || { echo "CUDA installation failed"; exit 1; }
fi

if [ ! -d "textured_gaussians" ]; then
    git clone --recurse-submodules https://github.com/Camak2441/textured_gaussians || { echo "Failed to clone textured_gaussians"; exit 1; }
fi
cd textured_gaussians || { echo "Failed to cd into textured_gaussians"; exit 1; }

mkdir -p data results

ENV_NAME="${ENV_NAME:-textured_gaussians}"

conda deactivate
conda env create --name "$ENV_NAME" --file=environment.yml || { echo "Failed to make conda env"; exit 1; }
conda activate "$ENV_NAME"
. scripts/setup.sh
pip install -e . || { echo "Failed to install textured_gaussians"; exit 1; }
