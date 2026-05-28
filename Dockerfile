FROM nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    PATH="/opt/venv/bin:$PATH"

ARG VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 git curl gnupg \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124 \
 && /opt/venv/bin/pip install git+https://github.com/cellarium-ai/cellarium-ml@${VERSION} \
 && rm -rf ~/.cache/pip