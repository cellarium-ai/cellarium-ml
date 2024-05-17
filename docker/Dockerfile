# Start from nvidia-docker image with drivers pre-installed to use a GPU
FROM nvcr.io/nvidia/cuda:12.4.1-base-ubuntu20.04

# Specify a certain commit via branch, tag, or sha
ARG GIT_SHA

LABEL maintainer="Stephen Fleming <sfleming@broadinstitute.org>"
ENV DOCKER=true \
    CONDA_AUTO_UPDATE_CONDA=false \
    CONDA_DIR="/opt/conda" \
    GIT_SHA=$GIT_SHA
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates sudo git \
 && apt-get clean \
 && sudo rm -rf /var/lib/apt/lists/* \
# get miniconda
 && curl -so $HOME/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh \
 && chmod +x $HOME/miniconda.sh \
 && $HOME/miniconda.sh -b -p $CONDA_DIR \
 && rm $HOME/miniconda.sh \
# install package and its dependencies
 && yes | pip install --no-cache-dir -U git+https://github.com/cellarium-ai/cellarium-ml.git@$GIT_SHA \
 && conda clean -yaf \
 && sudo rm -rf ~/.cache/pip
