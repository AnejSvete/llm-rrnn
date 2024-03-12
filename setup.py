from setuptools import setup

install_requires = [
    "dill",
    "frozendict",
    "numpy",
    "pytest",
    "scipy",
]


setup(
    name="rrnn",
    install_requires=install_requires,
    version="1.0",
    scripts=[],
    packages=["rrnn"],
)
