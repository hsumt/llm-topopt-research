# ParaLeSTO
ParaLeSTO is a Parallel Level Set Topology Optimization code using PETSc. See the [documentation](https://m2do1.gitlab.io/paralesto/) for information about functionality.

Note that this code has only been tested for Linux Ubuntu and may not be compatible with other operating systems.
For Windows users, it is recommended to download WSL (Windows Subsystem for Linux) from Microsoft, and then follow the Linux installation guides for the appropriate software dependencies.

## Software dependencies
### C++ dependencies
1. [PETSc](https://www.mcs.anl.gov/petsc/download/index.html) 3.18
2. [TopOpt in PETSc](https://github.com/topopt/TopOpt_in_PETSc)
3. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) 3.4.0
4. [NLOPT](https://nlopt.readthedocs.io/en/latest/) 2.7.1
5. [GLPK](https://www.gnu.org/software/glpk/) 5.0-1
### Python plugin dependencies
1. [anaconda](https://docs.anaconda.com/) 22.9.0
2. [numpy](https://numpy.org/) 1.23.3
3. [dolfin-adjoint](https://www.dolfin-adjoint.org/en/latest/) 2019.1.0
4. [scipy](https://scipy.org/) 1.9.1
4. [nlopt](https://nloypt.readthedocs.io/en/latest/) 2.7.1
5. [glpk](https://www.gnu.org/software/glpk/) 5.0
6. [matplotlib](https://matplotlib.org/) 3.6.1
7. [scikit-learn](https://scikit-learn.org/stable/) 1.2.1
8. [Cython](https://cython.readthedocs.io/en/latest/) 3.0.0

## Instructions for installing C++ dependencies
### PETSc
1. Clone PETSc using the command

        git clone -b release https://gitlab.com/petsc/petsc.git petsc

   (optional) Clone PETSc into the same parent directory as ParaLeSTO

2. Navigate to petsc folder and configure petsc using the command 

        ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack

   Note that the output of this command will show the `PETSC_DIR` and `PETSC_ARCH` paths, which will be needed for the example makefiles.
   Also, this command requires that `gcc`, `g++`, `gfortran`, and `make` be installed, which can be done on linux ubuntu using the `sudo apt install` command.
3. Follow instructions on the terminal that appear after configuring petsc for installation
  
### TopOpt in PETSc
1. This does not require installation. The repository, however, is available on github at https://github.com/topopt/TopOpt_in_PETSc.

### Eigen
1. Clone Eigen using the command

        git clone https://gitlab.com/libeigen/eigen.git

   (optional) Clone Eigen into the same parent directory as ParaLeSTO

2. Once everything is extracted, take note of where Eigen is in your library. This path will be called upon in your makefiles.

3. (optional) The website has some great tutorials on using Eigen, at https://eigen.tuxfamily.org/index.php?title=Main_Page

### NLopt
1. To clone the repository:

        git clone https://github.com/stevengj/nlopt.git

2. Navigate to the nlopt folder and follow the installation guide provided here: https://nlopt.readthedocs.io/en/latest/NLopt_Installation/
   Note that the installtion requires `cmake`, which can be installed on linux ubuntu using `sudo apt install cmake`

### GLPK
1. The tarbell can be downloaded from one of the many mirrors at:

   https://www.gnu.org/prep/ftp.html

2. Once extracted, the file 'INSTALL' contains instructions for installation.

## Instructions for installing Python dependencies
Note that this installation is only necessary for using ParaLeSTO from python.

### Anaconda
1. Follow the installation instructions from the [anaconda documentation](https://docs.continuum.io/anaconda/install/linux/).

### Installing all other dependencies in an anaconda enviroment
2. Open a terminal in the `paralesto` directory and create the `pyparalestoenv` environment

        conda env create -f environment.yml

   This will create an anaconda environment `pyparalestoenv` that has all the dependencies installed. Note that this step may take several minutes.

3. Enter the `pyparalestoenv` environment

        conda activate pyparalestoenv

### Install pyparalesto libraries
4. Use the following command

        pip install -e .

   To exit the anaconda environment, use the command `conda deactivate`

## Instructions for running ParaLeSTO
### Preparing the makefile
1. Navigate to an example, e.g. `paralesto/examples/agave`
2. Set up `PETSC_DIR` and `PETSC_ARCH` according to petsc installation
3. Set up EIGEN_DIR where Eigen is installed
4. Set up LIBOTHER where NLOPT and GLPK are installed

### Running
5. In `paralesto/examples/agave`, open a terminal and make the target using the following command

        make topopt

6. Run the program using the following command (replace `[PETSC_DIR]` and `[PETSC_ARCH]` with the appropriate path and `[NUM_CORES]` with the number of cores)

        [PETSC_DIR]/[PETSC_ARCH]/bin/mpiexec -n [NUM_CORES] ./topopt.out

   A sample command using 8 cores is shown below

        /home/m2do/petsc/arch-linux-c-debug/bin/mpiexec -n 8 ./topopt.out

## Instructions for running ParaLeSTO from python
1. Open a terminal and enter the anaconda environment using the following command
        
        conda activate pyparalestoenv
        
2. Navigate to a folder with a python example, e.g. `paralesto/examples/2022_paper_ex/ex2`, and run the example using the following command

        python main.py

## Viewing optimization results
Optimization results are exported to stl files. We recommend using [Paraview](https://www.paraview.org/) for visualization, though other applications can be used as well.

## Versions and releases
See the [release page](https://gitlab.com/m2dO1/paralesto/-/releases) for previous and latest releases. 

The versioning scheme used is [Semantic Versioning](https://semver.org/).

## Citation
To cite this package, please cite the following:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7613753.svg)](https://doi.org/10.5281/zenodo.7613753)

Note that this software is registered with [Zenodo](https://zenodo.org) 

## Contact
If you have any questions, create an issue or email m2do@ucsd.edu.
