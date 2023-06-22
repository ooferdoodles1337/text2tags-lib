from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name="text2tags-lib",
    version="0.0.5",
    license='MIT',
    description='Use a finetune of Llama-7b and llama cpp to predict Danbooru tags from natural text.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Pantai Suyasri',
    author_email='pantaisuyasri@gmail.com',
    url='https://github.com/DatboiiPuntai/text2tags-lib',
    download_url='https://github.com/DatboiiPuntai/text2tags-lib/archive/refs/heads/master.zip',
    packages=find_packages(),
    install_requires=["editdistance", "llama-cpp-python", "wget"],
    package_data={"text2tags": ["tags.txt"]},
)
