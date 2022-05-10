import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
   requirements = fh.readlines()
   requirements = [requirement.strip().replace('\n','').replace('\r','') for requirement in requirements]
   requirements = [requirement for requirement in requirements if len(requirement) != 0 and requirement[0] != '#']

setuptools.setup(
    name='doe_xstock',
    version='0.0.1',
    author='Kingsley Nweye',
    author_email='etonwana@yahoo.com',
    description='Manage DOE\'s End Use Load Profiles for the U.S. Building Stock.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intelligent-environments-lab/doe_xstock',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={'console_scripts': ['doe_xstock = doe_xstock.main:main']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)