from setuptools import setup, find_packages

package_name = 'task_grounding'

setup(
    name=package_name,
    version='0.0.0',
    packages=["task_grounding"],
    install_requires=['setuptools',
                        'numpy',
                        'sqlite3'],
    zip_safe=True,
    maintainer='',
    description='Task grounding node',
    license='TODO: License declaration',
    tests_require=[],
)
