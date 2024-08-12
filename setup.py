from setuptools import setup, find_packages

setup(
        name='nnuebit',
        version='0.0a0',
        scripts=['bin/nnuebit', 'bin/visbit', 'bin/quantize', 'bin/evaluatennue'],
        packages=find_packages(),
        author='Isak Ellmer',
        author_email='isak01@gmail.com',
        description='An nnue trainer for bitbit.',
        url='https://github.com/spinojara/nnuebit',
)
