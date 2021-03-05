from setuptools import setup, find_packages

package_name = 'grounding'

setup(
    name=package_name,
    version='0.0.0',
    packages=["grounding"],
    install_requires=['setuptools',
                        'numpy,'
                        'sqlite3'],
    zip_safe=True,
    maintainer='',
    description='Grounding node',
    license='TODO: License declaration',
    tests_require=[],
)
