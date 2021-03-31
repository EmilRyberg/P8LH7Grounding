from setuptools import setup, find_packages

package_name = 'ner_lib'

setup(
    name=package_name,
    version='0.0.1',
    packages=["ner_lib"],
    install_requires=['setuptools',
                        'torch',
                        'numpy'],
    maintainer='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=[]
)