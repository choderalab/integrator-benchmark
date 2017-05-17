from setuptools import setup

setup(name='benchmark',
      version='0.0.0',
      description='Enumerating and evaluating numerical integrators of Langevin dynamics',
      url='https://github.com/choderalab/integrator-benchmark',
      author='Josh Fass, John Chodera',
      author_email='{josh.fass, john.chodera}@choderalab.org',
      license='MIT',
      packages=['benchmark', 'benchmark.evaluation', 'benchmark.experiments', 'benchmark.integrators',
                'benchmark.plotting', 'benchmark.serialization', 'benchmark.tests', 'benchmark.testsystems',
                'benchmark.utilities', 'benchmark.verification'])
