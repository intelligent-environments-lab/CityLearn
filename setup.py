import os
import re
import setuptools

ROOT = os.path.dirname(__file__)
VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([\w\.-]+)['\"]")
DIST_NAME = os.getenv('CITYLEARN_PYPI_NAME', 'softcpsrecsimulator')

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as fh:
   requirements = fh.readlines()
   requirements = [requirement.strip().replace('\n','').replace('\r','') for requirement in requirements]
   requirements = [requirement for requirement in requirements if len(requirement) != 0 and requirement[0] != '#']

def get_version():
    init = open(os.path.join(ROOT, 'citylearn', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)

setuptools.setup(
    name=DIST_NAME,
    version=get_version(),
    author='Soft-CPS Research Group, Jose Ramon Vazquez-Canteli, Kingsley Nweye, Zoltan Nagy',
    author_email='jose@isep.ipp.pt',
    description=(
        'An open source Farama Foundation Gymnasium environment for benchmarking distributed '
        'energy resource control algorithms to provide energy flexibility in a district of buildings.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Soft-CPS-Research-Group/Simulator',
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={'console_scripts': ['citylearn = citylearn.__main__:main']},
    project_urls={
        'Source': 'https://github.com/Soft-CPS-Research-Group/Simulator',
        'Original Project': 'https://github.com/intelligent-environments-lab/CityLearn',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
