from setuptools import setup, find_packages


setup(name='KCEF',
      version='0.1',
      description=' Estimator for kernel conditional exponential family model',
      url='git@bitbucket.org:MichaelArbel/KCEF.git',
      author='Michael Arbel',
      author_email='michael.n.arbel@gmail.com',
      license='BSD3',
      packages=find_packages('.', exclude=["*tests*", "*.develop"]),
      zip_safe=False)
