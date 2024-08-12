from setuptools import setup, find_packages

setup(
        name='nnuebit',
        version='0.0a0',
        scripts=['bin/trainbit', 'bin/visbit', 'bin/quantbit', 'bin/evalnnue'],
        packages=find_packages(),
        author='Isak Ellmer',
        author_email='isak01@gmail.com',
        description='An nnue trainer for bitbit.',
        url='https://github.com/spinojara/nnuebit',
        install_requires=[
            'torch',
            'matplotlib',
            'numpy',
        ],
)
