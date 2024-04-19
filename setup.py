from setuptools import setup, find_packages

setup(
  name = 'torch-afem',
  packages=find_packages(include=['torch_afem', 'torch_afem.*']),
  version = '0.0.1',
  license='MIT',
  description = 'PyTorch Finite Element Method',
  long_description='PyTorch Finite Element Method',
  long_description_content_type="text/markdown",
  author = 'Shuhao Cao',
  author_email = 'scao.math@gmail.com',
  url = 'https://github.com/scaomath/torch-fem',
  keywords = ['pytorch', 'fem', 'pde'],
  install_requires=[
      'seaborn',
      'torchinfo',
      'numpy',
      'torch>=1.9.0',
      'plotly',
      'scipy',
      'psutil',
      'matplotlib',
      'tqdm',
      'PyYAML',
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.8',
  ],
)
