from setuptools import setup, find_packages
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__), 'requirements.txt'))):
    with open(requirement_path) as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name='aerobot',
    version='0.0.1',
    packages=find_packages(where='aerobot'),
    entry_points={'console_scripts':[
        'download==aerobot.cli:download',
        'train==aerobot.cli:download',
        'predict==aerobot.cli:download']},
    package_data={'aerobot' :[]},
    include_package_data=True,
    install_requires=get_requirements(),
    py_modules=[],
    python_requires='>=3.8')
