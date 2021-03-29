from setuptools import setup, find_packages

package_name = 'feature_extractor'

setup(
    name=package_name,
    version='0.0.1',
    packages=["feature_extractor"],
    install_requires=['setuptools',
                        'torch',
                        'numpy'],
    maintainer='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=[]
)