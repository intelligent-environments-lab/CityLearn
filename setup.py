import os
import re
import setuptools

ROOT = os.path.dirname(__file__)
VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([\w\.-]+)['\"]")

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
   requirements = fh.readlines()
   requirements = [requirement.strip().replace('\n','').replace('\r','') for requirement in requirements]
   requirements = [requirement for requirement in requirements if len(requirement) != 0 and requirement[0] != '#']

def get_version():
    init = open(os.path.join(ROOT, 'citylearn', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)

setuptools.setup(
    name='CityLearn',
    version=get_version(),
    author='Jose Ramon Vazquez-Canteli, Kingsley Nweye, Zoltan Nagy',
    author_email='nweye@utexas.edu',
    description='An open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intelligent-environments-lab/CityLearn',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={'console_scripts': ['citylearn = citylearn.main:main']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)