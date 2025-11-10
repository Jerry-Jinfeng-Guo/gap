#!/usr/bin/env python3
"""
setup.py for GAP Python Bindings

This setup script builds the GAP Power Flow Calculator Python bindings using pybind11.
Supports both CPU and GPU (CUDA) backends with automatic detection.
"""

import os
import sys
import subprocess
import multiprocessing
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    """A Python extension built using CMake"""
    
    def __init__(self, name, cmake_dir='.'):
        super().__init__(name, sources=[])
        self.cmake_dir = os.path.abspath(cmake_dir)


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake to build extensions"""
    
    def run(self):
        # Check if CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build GAP Python bindings")
            
        for ext in self.extensions:
            self.build_cmake_extension(ext)
    
    def build_cmake_extension(self, ext):
        """Build a CMake extension"""
        ext_fullpath = self.get_ext_fullpath(ext.name)
        ext_filename = os.path.basename(ext_fullpath)
        ext_dir = os.path.dirname(ext_fullpath)
        
        # Create build directory
        build_dir = os.path.join(self.build_temp, 'cmake_build')
        os.makedirs(build_dir, exist_ok=True)
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DGAP_BUILD_PYTHON_BINDINGS=ON',
        ]
        
        # Build configuration
        build_mode = 'Debug' if self.debug else 'Release'
        cmake_args.append(f'-DCMAKE_BUILD_TYPE={build_mode}')
        
        # Platform-specific arguments
        if sys.platform.startswith('win'):
            cmake_args.extend([
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{build_mode.upper()}={ext_dir}',
                f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{build_mode.upper()}={ext_dir}'
            ])
        
        # Check for CUDA availability
        cuda_available = self.check_cuda_availability()
        if cuda_available:
            print("CUDA detected - enabling GPU support")
            cmake_args.append('-DGAP_ENABLE_CUDA=ON')
        else:
            print("CUDA not found - building CPU-only version")
            cmake_args.append('-DGAP_ENABLE_CUDA=OFF')
        
        # Number of parallel build jobs
        build_jobs = os.environ.get('GAP_BUILD_JOBS', str(multiprocessing.cpu_count()))
        
        # CMake build arguments
        build_args = ['--config', build_mode]
        if sys.platform.startswith('win'):
            build_args.extend(['--', f'/m:{build_jobs}'])
        else:
            build_args.extend(['--', f'-j{build_jobs}'])
        
        # Configure
        print(f"Configuring CMake with args: {cmake_args}")
        subprocess.check_call(['cmake', ext.cmake_dir] + cmake_args, cwd=build_dir)
        
        # Build
        print(f"Building with args: {build_args}")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_dir)
    
    def check_cuda_availability(self):
        """Check if CUDA is available on the system"""
        # Check for CUDA environment variables
        cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
        if cuda_path and os.path.exists(cuda_path):
            return True
        
        # Check for nvcc compiler
        try:
            subprocess.check_output(['nvcc', '--version'], stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.CalledProcessError):
            pass
        
        # Check common CUDA installation paths
        common_cuda_paths = [
            '/usr/local/cuda',
            '/opt/cuda',
            '/usr/lib/nvidia-cuda-toolkit',
        ]
        
        for path in common_cuda_paths:
            if os.path.exists(os.path.join(path, 'bin', 'nvcc')):
                return True
        
        return False


def get_long_description():
    """Get the long description from README file"""
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return "GAP: GPU-Accelerated Power Flow Calculator - Python Bindings"


if __name__ == '__main__':
    setup(
        name='gap-solver',
        version='1.0.0',
        author='GAP Development Team',
        author_email='gap@powerflow.com',
        description='GAP: GPU-Accelerated Power Flow Calculator - Python Bindings',
        long_description=get_long_description(),
        long_description_content_type='text/markdown',
        url='https://github.com/Jerry-Jinfeng-Guo/gap',
        
        # Python package configuration
        packages=find_packages(),
        package_dir={'gap': 'bindings'},
        
        # C++ extension configuration
        ext_modules=[CMakeExtension('gap.gap_solver')],
        cmdclass={'build_ext': CMakeBuild},
        
        # Requirements
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.19.0',
            'scipy>=1.6.0',
        ],
        
        extras_require={
            'dev': [
                'pytest>=6.0',
                'pytest-benchmark',
                'black',
                'flake8',
                'mypy',
            ],
            'validation': [
                'power-grid-model>=1.9.0',
                'matplotlib>=3.3.0',
                'pandas>=1.2.0',
            ],
        },
        
        # Metadata
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: C++',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
        ],
        keywords='power flow power system newton-raphson gpu cuda electrical engineering',
        license='MIT',
        
        # Include package data
        include_package_data=True,
        zip_safe=False,
    )