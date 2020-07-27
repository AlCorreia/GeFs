from setuptools import setup

setup(name='gefs',
      version='0.1',
      description='Generative Forests',
      url='http://github.com/alcorreia/gefs',
      author='Alvaro H. C. Correia',
      author_email='a.h.chaim.correia@tue.nl',
      license='MIT',
      packages=['gefs'],
      install_requires=['numpy', 'numba', 'pandas', 'scipy', 'sklearn', 'tqdm'],
      zip_safe=False)
