from setuptools import setup, find_packages


setup(
    name="text2tags-lib",
    version="0.0.2",
    license='MIT',
    description='Use a finetune of Llama-7b and llama cpp to predict Danbooru tags from natural text.',
    author='Pantai Suyasri',
    author_email='pantaisuyasri@gmail.com',
    url='https://github.com/DatboiiPuntai/text2tags-lib',
    download_url='https://github.com/DatboiiPuntai/text2tags-lib/archive/refs/tags/v0.0.1.zip',
    packages=find_packages(),
    install_requires=["editdistance", "llama-cpp-python==1.5.0", "wget"],
    package_data={"text2tags": ["tags.txt"]},
)
