from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'LSH Recommendation System Package'
LONG_DESCRIPTION = 'A Python package implementing a Locality Sensitive Hashing (LSH) recommendation system.'

# Setting up
setup(
    name="lsh_recommender",
    version=VERSION,
    author="Mingjia Guan",
    author_email="mguan@stu.feitian.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["nltk", "numpy", "scipy", "scikit-learn"],
    keywords=['recommendation system', 'LSH', 'MinHashing'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ]
)
