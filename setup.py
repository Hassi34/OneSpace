import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "onespace"
USERNAME = "Hassi34"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    version="0.0.2",
    author=USERNAME,
    author_email="hasanain@aicaliber.com",
    description="A high-level Python framework to automate the project lifecycle of Machine and Deep Learning Projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USERNAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "vizpool>=0.0.9",
        "tensorflow==2.8.2",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "tqdm==4.64.1",
        "xgboost==1.6.2",
        "imbalanced-learn==0.8.1",
        "lightgbm==3.3.2",
        "python-dotenv==0.19.2",
        "pymongo[srv]==4.2.0",
        "PyMySQL==1.0.2",
        "SQLAlchemy==1.4.41"
    ]
)
