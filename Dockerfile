FROM nvcr.io/nvidia/cuda:13.2.0-base-ubuntu24.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true

ARG VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install git+https://github.com/cellarium-ai/cellarium-ml@${VERSION} \
 && rm -rf ~/.cache/pip
 