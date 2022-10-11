<p align="center">
<b><h1 style="text-align:center"><em>♾️ OneSpace ♾️</em></h1></b>
</p>
<p align="center">
    <em>A high-level Python framework to automate the project lifecycle of Machine and Deep Learning Projects</em>
</p>

[![PyPI Version](https://img.shields.io/pypi/v/onespace?color=g)](https://pypi.org/project/onespace)
[![Python Version](https://img.shields.io/pypi/pyversions/onespace?color=g)](https://pypi.org/project/onespace)
[![Downloads](https://static.pepy.tech/personalized-badge/onespace?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/onespace)
[![Last Commit](https://img.shields.io/github/last-commit/hassi34/onespace/main?color=g)](https://github.com/hassi34/onespace)
[![License](https://img.shields.io/github/license/hassi34/onespace?color=g)](https://pypi.org/project/onespace)
[![Issue Tracking](https://img.shields.io/badge/issue_tracking-github-brightgreen.svg)](https://github.com/hassi34/onespace/issues)
[![Open Issues](https://img.shields.io/github/issues/hassi34/onespace)](https://github.com/hassi34/onespace/issues) 
[![Github Actions Status](https://img.shields.io/github/workflow/status/hassi34/onespace/Publish%20Python%20distributions%20to%20PyPI%20and%20TestPyPI?event=push)](https://pypi.org/project/onespace)
[![Code Size](https://img.shields.io/github/languages/code-size/hassi34/onespace?color=g)](https://pypi.org/project/onespace)
[![Repo Size](https://img.shields.io/github/repo-size/hassi34/onespace?color=g)](https://pypi.org/project/onespace)

## Overview
OneSpace enables you to train high performace production ready Machine Learning And Deep Learning Models effortlessly with less than five lines of code. ``OneSpace`` provides you a unified workspace so you can work on all kinds of Machine Learning and Deep Learning problem statements without having to leave your workspace environment. OneSpace don't just run everything as a black box to present you results, instead it makes the complete model training process easily explainable using artifacts, logs and plots. OneSpace also provides you the optional parameters to pass database credentials which will create a table with the project name in the database of your choice and will log all the training activities and the parameters in the database.<br>
Following are the major contents to follow, you can jump to any section:

>   1. [Installation](#install-)
>   2. [Usage](#use-)
>   3. [Getting Started with OneSpace (Tutorials)](#tutorials-)<br>
    A ) - [Tabular](#tabular-)<br>
    i ) - [Training a Regression Model](#reg-)<br>
    ii) - [Training a Classification Model](#reg-)<br>
    B ) - [Computer Vision](#vision-)<br>
    i ) - [Training an Image Classification Model with Tensorflow](#tf-imgcls)<br>
    ii) - [Training an Image Classification Model with PyTorch](#pytorch-imgcls)<br>
>   4. [Contributing](#contributing-)
### 🔗 Project Link
**``OneSpace``** is being distributed through PyPI. Check out the PyPI Package [here](https://pypi.org/project/onespace/)


### 1. Installation<a id='install-'></a>
To avoid any dependency conflict, make sure to create a new Python virtual environment and then install via Pip!
```bash
pip install onespace
```
### 2. Usage<a id='use-'></a>
Get the **[config.py](https://github.com/Hassi34/onespace/blob/main/tabularConfig.py)** and **[training.py](https://github.com/Hassi34/onespace/blob/main/training.py)** files ready. You can get it from this repo or from the following tutorials section. 
* Prepare ``training.py``
```bash
import config # Incase, you renamed config.py to something else, make sure to use the same name here
from src.onespace.tabular.regression import Experiment # Importing Experiment class to train a regression model

def training(config):
    exp = Experiment(config)
    exp.run_experiment()

if __name__ == "__main__":
    training(config)
```
* Now run the following command on your terminal to start the training job:
```bash
python training.py
```
Please following along with these ``quick tutorials``👇 to understand the complete setup and training process.
### 3. Getting Started with OneSpace<a id='tutorials-'></a>

* Ensure you have [Python 3.7+](https://www.python.org/downloads/) installed.

* Create a new Python conda environment for the OneSpace:

```
$ conda create -n venv  # create venv
$ conda activate venv  # activate venv
$ pip install onespace # install onespace
```

* Create a new Python virtual environment with pip for the OneSpace:
```
$ python3 -m venv venv  # create venv
$ . venv/bin/activate   # activate venv
$ pip install onespace # install onespace
```

### 4. Contributing<a id='contributing-'></a>
Yes, Please! We believe that there is alot of oportunity to make Machine Learning more interesting and less complicated for the comunity, so let's make it more efficient, let's go with low-code!!

#### **Please give this repository a star if you find our work useful, Thank you! 🙏**<br><br>

**Copyright &copy; 2022 onespace** <br>
Let's connect on **[``LinkedIn``](https://www.linkedin.com/in/hasanain-mehmood-a37a4116b/)** <br>

