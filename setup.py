from setuptools import setup, find_packages
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name='aerobot',
    version='0.0.1',
    entry_points={'console_scripts':[
        'download=aerobot.cli:download',
        'train=aerobot.cli:train',
        'models=aerobot.cli:models',
        'embed=aerobot.cli:embed',
        'predict=aerobot.cli:predict']},
    install_requires=get_requirements(),
    packages=['aerobot', 'aerobot.data', 'aerobot.models'],
    python_requires='>=3.8')
