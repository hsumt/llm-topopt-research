FROM dolfinx/dolfinx:stable
LABEL Author="M. Hsu"

USER root
ARG DEBIAN_FRONTEND=noninteractive

ENV DIRPATH=/usr/local/src
WORKDIR $DIRPATH
RUN apt-get update -y && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git-all \
    htop \
    nano \
    wget \
    sudo \
 && apt-get clean -y \
 && ln -s /usr/bin/python3 /usr/bin/python
# Install extra Python packages
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
    ipykernel \
    jsonschema \
    gmsh \
    numpy \
    matplotlib \
    pandas \
    scipy \
    pydantic \
    anthropic \
    h5py \
    python-dotenv \
    meshio \
    openmdao
 