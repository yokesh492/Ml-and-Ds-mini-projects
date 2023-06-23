from setuptools import find_packages,setup
from typing import List

flag = '-e .'
def get_requirements(file_path:str) ->List[str]:
    requirements =[]
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if flag in requirements:
            requirements.remove(flag)
    return requirements

setup(
    name = 'digiverz_task1',
    version = '0.0.1',
    author = 'yogesh k',
    author_email= 'yokeshkummar@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt'),
)