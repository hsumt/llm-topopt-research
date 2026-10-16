# Pinned conda environment with FEniCSx (DOLFINx 0.7.0), not legacy FEniCS.
FROM anaconda/miniconda@sha256:ffb09b25d6ba331b18f0a87f5b2f0fc1b6e661f99f765e4c2d7f61e015df0efd

WORKDIR /app
# ParaView is mounted from the host at /opt/paraview (docker-compose.yml). These are the system
# libraries its official Linux binary needs for headless OSMesa rendering in this Debian image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgomp1 libpciaccess0 libx11-6 libxext6 libxt6 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN conda create --yes --override-channels --channel conda-forge --prefix /opt/ato --file /tmp/conda-linux-64.lock \
    && conda clean -afy

ENV PATH=/opt/ato/bin:$PATH \
    CONDA_PREFIX=/opt/ato \
    CC=/opt/ato/bin/x86_64-conda-linux-gnu-cc \
    CXX=/opt/ato/bin/x86_64-conda-linux-gnu-c++ \
    CPPFLAGS="-isystem /opt/ato/include -isystem /opt/ato/include/eigen3" \
    CXXFLAGS="-std=c++14 -isystem /opt/ato/include -isystem /opt/ato/include/eigen3" \
    LDFLAGS="-L/opt/ato/lib -Wl,-rpath,/opt/ato/lib" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Compile inside the image; no host binary, checkout, or editable-install path.
COPY _external /app/_external
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps /app/_external/pyparalesto \
    && python -c "from mpi4py import MPI; import dolfinx; from pyparalesto.pylsm2d import PyInput, PyLevelSetModule; from pyparalesto.pyopt import PyOptimizerModule; assert dolfinx.__version__ == '0.7.0'"

RUN mkdir -p /home/ato /app/artifacts && chmod 1777 /home/ato /app/artifacts
ENV HOME=/home/ato XDG_CACHE_HOME=/home/ato/.cache MPLCONFIGDIR=/home/ato/.config/matplotlib
COPY project /app/project
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "project/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
