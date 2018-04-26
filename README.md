## Torchlite
![](./docs/sources/img/logo.png)
[![PyPI version](https://badge.fury.io/py/torchlite.svg)](https://badge.fury.io/py/torchlite)

Torchlite is a high level library on top of popular machine learning frameworks such as
sklearn, Pytorch and Tensorflow.
It gives a high layer abstraction of repetitive code used in machine learning for day-to-day data science tasks.

## Installation

```
pip install torchlite
```

or if you want to run this lib directly to have access to the examples clone this repository and run:

```
pip install -r requirements.txt
```

to install the required dependencies.
By default **Pytorch 0.4.0+** and **Tensorflow-GPU 1.8.0+** are installed along with this library but it's recommended
to install them from source from [here](http://pytorch.org/) if you want to use the `torchlite.torch`
package and/or head over to the [Tensorflow install page](https://www.tensorflow.org/install/) if you want to
use the `torchlite.tf` package.

## Documentation

For now the library has no complete documentation but you can quickly get to know how
it works by looking at the examples in the `examples-*` folders. This library is still in
alpha and few APIs may change in the future. The only things which will evolve at the same
pace as the library are the examples, they are meant to always be up to date with
the library.

Few examples will generates folders/files such as saved models or tensorboard logs.
To visualize the tensorboard logs download Tensorflow's tensorboard as well as 
[Pytorch's tensorboard](https://github.com/lanpa/tensorboard-pytorch) if you're interested by
the `torchlite.torch` package. Then execute:
```
tensorboard --logdir=./tensorboard
```

## Packaging the project for Pypi deploy

```
pip install twine
pip install wheel
python setup.py sdist
python setup.py bdist_wheel
```

[Create a pypi account](https://packaging.python.org/tutorials/distributing-packages/#id76) and create `$HOME/.pypirc` with:
```
[pypi]
username = <username>
password = <password>
```

Then upload the packages with:
```
twine upload dist/*
```

Or just:
```
pypi_deploy.sh
```
