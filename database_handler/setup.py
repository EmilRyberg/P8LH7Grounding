from setuptools import setup, find_packages

package_name = 'database_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=["database_handler"],
    install_requires=['setuptools',
                        'numpy',
                        'sqlite3'],
    zip_safe=True,
    maintainer='',
    description='Database handler package',
    license='TODO: License declaration',
    tests_require=[],
)
