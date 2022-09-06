from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'Automate Linear Regression'
LONG_DESCRIPTION = 'A package that allows to build a Linear Regression Machine Learning model provided that the user has already surmised that the data the user has been working on follows a linear relationship. In addition regularization techniques can be implemented with the help of this package.'

# Setting up
setup(
    name="vidstream",
    version=VERSION,
    author="Suryansh Grover",
    author_email="<suryanshgrover1999@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scikit-learn', 'pandas', 'numpy', 'pickle'],
    keywords=['python', 'machine learning', 'machine learning model', 'regression', 'linear regression', 'lasso linear regression', 'elasticnet linear regression', 'regularization'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X"
    ]
)