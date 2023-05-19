from setuptools import setup, find_packages
from setuptools.command.install import install
from text2tags.functions import download_model

class DownloadModelCommand(install):
    def run(self):
        # Download the model file
        download_model()

setup(
    name="text2tags-lib",
    version="0.0",
    packages=find_packages(),
    install_requires=["editdistance", "llama-cpp-python", "wget"],
    cmdclass={"install": DownloadModelCommand},
)
