from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import glob
import os

# Use os to get the user's home directory
home = os.path.expanduser("~")

# Define sources for level set extension
lsm_sources = glob.glob("lsm/src/*.cpp")
lsm_sources.remove("lsm/src/optimizer_wrapper.cpp")
lsm_sources.remove("lsm/src/sensitivity.cpp")
lsm_sources.remove("lsm/src/optimize_ipopt.cpp")
lsm_sources.append("pyparalesto/py_lsm/initialize_lsm.cpp")
lsm_sources.append("pyparalesto/py_lsm/py_lsm_cy.pyx")

# Define sources for 2d level set extension
lsm_2d_sources = glob.glob("lsm_2d/src/*.cpp")
lsm_2d_sources.append("pyparalesto/py_lsm_2d/initialize_lsm.cpp")
lsm_2d_sources.append("pyparalesto/py_lsm_2d/py_lsm_2d_cy.pyx")

# Define sources for optimizer extension
opt_sources = ["lsm/src/optimize.cpp",
               # "lsm/src/optimize_ipopt.cpp",
               "lsm/src/optimizer_wrapper.cpp",
               "pyparalesto/py_opt/py_opt_cy.pyx",]

# Setup pylsm and py_opt extenstions
ext_modules = [
    Extension(
        "pylsm", # extension name
        sources            = lsm_sources, # C++ source files and pyx file
        libraries          = ["nlopt", "glpk"], # "ipopt"],
        include_dirs       = [np.get_include(),
                              "pyparalesto/py_lsm/", "lsm/include",
                              home, # folder where eigen3 is located
                              "/usr/local/include/",],
        library_dirs       = ["/usr/", "/usr/lib", "/usr/local/lib/",
                              "/lsm/include/",],
        extra_compile_args = ["-std=c++14", "-D PYBIND", "-g", "-pthread",
                              "-Wno-sign-compare"],
        extra_link_args    = ["-g"],
        language           = 'c++',
        ),
    Extension(
        "pylsm2d", # extension name
        sources            = lsm_2d_sources, # C++ source files and pyx file
        libraries          = [],
        include_dirs       = [np.get_include(),
                              "pyparalesto/py_lsm_2d/", "lsm_2d/include",
                              home, # folder where eigen3 is located
                              "/usr/local/include/",],
        library_dirs       = ["/usr/", "/usr/lib", "/usr/local/lib/",
                              "/lsm_2d/include/",],
        extra_compile_args = ["-std=c++14", "-D PYBIND", "-g", "-pthread",
                              "-Wno-sign-compare"],
        extra_link_args    = ["-g"],
        language           = 'c++',
        ),
    Extension(
        "pyopt", # extension name
        sources            = opt_sources, # C++ source files and pyx file
        libraries          = ["nlopt", "glpk"], # "ipopt"],
        include_dirs       = [np.get_include(),
                              "pyparalesto/py_opt/", "lsm/include",
                              home, # folder where eigen3 is located
                              "/usr/local/include/"],
        library_dirs       = ["/usr/", "/usr/lib", "/usr/local/lib/",
                              "/lsm/include/",],
        extra_compile_args = ["-std=c++14", "-D PYBIND", "-g", "-pthread",
                              "-Wno-sign-compare"],
        extra_link_args    = ["-g"],
        language           = 'c++',
        ),
    ]

pkgname = "pyparalesto"

setup(    
    name=pkgname,
    version="1.0.0", #different versioning than overall ParaLeSTO?
    author="Carolina Jauregui",
    author_email="cjauregu@eng.ucsd.edu",
    description="python package for ParaLeSTO",
    url="https://gitlab.com/m2dO1/paralesto",
    classifiers="Programming Language :: Python :: 3",
    license="Apache 2.0",
    license_files="LICENSE",
    platforms="Linux Ubuntu",
    packages=find_packages(include=['pyparalesto*']),
    cmdclass={"build_ext": build_ext},
    ext_package='pyparalesto',
    ext_modules = cythonize(ext_modules, language_level="3")
    )