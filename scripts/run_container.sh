#!/bin/bash
# Launch the SLIME training container with project files and Docker socket mounted.
# Run this from the project root on the host machine.

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

MSYS_NO_PATHCONV=1 docker run -it --rm \
    --gpus all \
    --shm-size 16g \
    --network host \
    -v "$PROJECT_DIR":/workspace/project \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -w /workspace/project \
    slimerl/slime:latest \
    bash
