FROM nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    PATH="/opt/venv/bin:$PATH"

ARG VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install git+https://github.com/cellarium-ai/cellarium-ml@${VERSION} \
 && rm -rf ~/.cache/pip