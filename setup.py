from setuptools import find_packages, setup

setup(
    name="erdos-sim",
    version="0.0.1",
    author="Pylot Team",
    description=("A simulator for emulating AV workloads."),
    long_description=open("README.md").read(),
    url="https://github.com/erdos-project/erdos-sim",
    keywords=("simulator for scheduling traces"),
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=["absl-py", "pyboolector", "z3"],
)
