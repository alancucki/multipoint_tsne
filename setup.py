from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


# Use obsolete _barnes_hut_tsne from sklearn
sklearn_tsne = Extension(
    '_barnes_hut_tsne', language='c', sources=['_barnes_hut_tsne.pyx'],
    extra_compile_args=["-Ofast", "-march=native", 
                        "-funroll-loops", "-Wno-unused-result"])
multipoint_tsne = Extension(
    '_barnes_hut_mptsne', language='c', sources=['_barnes_hut_mptsne.pyx'],
    extra_compile_args=["-Ofast", "-march=native", 
                        "-funroll-loops", "-Wno-unused-result"])

setup(ext_modules=cythonize([sklearn_tsne, multipoint_tsne]))
