from setuptools import setup, find_packages

setup(
    name="quantum_classical_hybrid",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'torch>=1.9.0',
        'matplotlib>=3.4.0',
        'pytest>=6.2.0',
        'numba>=0.54.0',
        'h5py>=3.3.0',
        'tqdm>=4.62.0',
    ],
    author="Naishal Patel",
    description="A quantum-classical hybrid simulator with neural quantum states",
    python_requires=">=3.8",
)