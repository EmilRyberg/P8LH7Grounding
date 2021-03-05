from setuptools import setup, find_packages

package_name = 'ner'

setup(
    name=package_name,
    version='0.0.1',
    packages=["ner"],
    install_requires=['setuptools',
                        'torch',
                        'numpy'],
    maintainer='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=[]
)