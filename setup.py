from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='medimm',
    version='0.0.1',
    description='PyTorch medical image models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mishgon/medimm',
    author='M. Goncharov',
    author_email='Mikhail.Goncharov2@skoltech.ru',
    packages=find_packages(include=('medimm',)),
    python_requires='>=3.6',
    install_requires=requirements,
)