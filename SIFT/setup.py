from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess

# 获取 OpenCV 的编译标志和链接标志
opencv_cflags = subprocess.getoutput('pkg-config --cflags opencv4').split()
opencv_libs = subprocess.getoutput('pkg-config --libs opencv4').split()

ext_modules = [
    Extension(
        'image_stitching',
        ['image_stitching_bindings.cpp', 'image_stitching.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ] + [flag[2:] for flag in opencv_cflags],  # 去掉 '-I' 前缀
        library_dirs=[],
        libraries=[lib[2:] for lib in opencv_libs if lib.startswith('-l')],
        extra_compile_args=[flag for flag in opencv_cflags if not flag.startswith('-I')],
        extra_link_args=[flag for flag in opencv_libs if not flag.startswith('-l')],
        language='c++'
    ),
]

setup(
    name='image_stitching',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='Image stitching module',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)


