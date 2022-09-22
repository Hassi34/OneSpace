import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "onespace"
USERNAME = "Hassi34"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    version="0.0.0",
    author= USERNAME,
    author_email="hasnainmehmood3435@gmail.com",
    description="A high-level python Framework with low-code support for Machine and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USERNAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires = [
        "tensorflow == 2.8.2",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "pandas >= 1.3.5",
        "tqdm==4.64.1",
        "matplotlib==3.5.3",
        "scikit-learn==1.0.2",
        "statsmodels==0.13.2",
        "xgboost==1.6.2",
        "imbalanced-learn==0.8.1",
        "seaborn==0.11.2",
        "lightgbm==3.3.2",
        "PyYAML==6.0"
        ]
)   