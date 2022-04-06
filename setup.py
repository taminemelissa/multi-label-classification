from setuptools import setup, find_packages

setup(name='multi-label-classification',
      version='0.0.1',
      description='',
      url='https://github.com/taminemelissa/multi-label-classification',
      author='Zineb Bentires, Sirine Louati, Louise Sirven, Mélissa Tamine',
      author_email='zineb.bentires@ensae.fr, sirine.louati@ensae.fr, louise.sirven@ensae.fr, melissa.tamine@ensae.fr',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'seaborn', 'torch', 'tqdm'])