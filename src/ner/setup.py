from setuptools import setup, find_packages

package_name = 'ner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools',
                        'torch',
                        'opencv-python',
                        'numpy'],
    zip_safe=True,
    maintainer='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=[],
)