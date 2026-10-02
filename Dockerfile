# Same clean-image pattern as mfrto; FEniCSx rather than legacy FEniCS.
FROM anaconda/miniconda@sha256:ffb09b25d6ba331b18f0a87f5b2f0fc1b6e661f99f765e4c2d7f61e015df0efd

WORKDIR /app
COPY docker/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN conda create --yes --override-channels --channel conda-forge --prefix /opt/lbracket --file /tmp/conda-linux-64.lock \
    && conda clean -afy

ENV PATH=/opt/lbracket/bin:$PATH \
    CONDA_PREFIX=/opt/lbracket \
    CC=/opt/lbracket/bin/x86_64-conda-linux-gnu-cc \
    CXX=/opt/lbracket/bin/x86_64-conda-linux-gnu-c++ \
    CPPFLAGS="-isystem /opt/lbracket/include -isystem /opt/lbracket/include/eigen3" \
    CXXFLAGS="-std=c++14 -isystem /opt/lbracket/include -isystem /opt/lbracket/include/eigen3" \
    LDFLAGS="-L/opt/lbracket/lib -Wl,-rpath,/opt/lbracket/lib" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Compile inside the image; no host binary, checkout, or editable-install path.
COPY _external /app/_external
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps /app/_external/pyparalesto \
    && python -c "from mpi4py import MPI; import dolfinx; from pyparalesto.pylsm import PyInput; from pyparalesto.pyopt import PyOptimizerModule; assert dolfinx.__version__ == '0.7.0'"

RUN mkdir -p /home/lbracket /app/artifacts && chmod 1777 /home/lbracket /app/artifacts
ENV HOME=/home/lbracket XDG_CACHE_HOME=/home/lbracket/.cache MPLCONFIGDIR=/home/lbracket/.config/matplotlib
COPY project /app/project
COPY tests /app/tests
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "project/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
