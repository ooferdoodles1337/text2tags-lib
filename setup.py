from setuptools import setup, find_packages


setup(
    name="text2tags-lib",
    version="0.0",
    packages=find_packages(),
    install_requires=["editdistance", "llama-cpp-python", "wget"],
    data_files=[('tags', ['text2tags/tags.txt'])],
)
