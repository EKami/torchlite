## High level library for Pytorch

Torchlite is a high level library meant to be what Keras is for Tensorflow and Theano.
It is not meant to micmic the Keras API at 100% but instead to get the best of both
worlds (Pytorch and Keras API). 
For instance if you used Keras train/validation generators, in Torchlite you would
use Pytorch [Dataset](http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset) and
[DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).

## Installation

```
pip install torchlite
```

or if you want to run this lib directly to have access to the examples clone this repository and run:

```
pip install -r requirements.txt
```

to install the required dependencies.
Then install pytorch and torchvision from [here](http://pytorch.org/).
Finally install the latest version of imgaug with:
```
pip install git+https://github.com/aleju/imgaug
```
Torchlite will use an outdated version from pypi by default.

## Documentation

For now the library has no complete documentation but you can quickly get to know how
it works by looking at the examples in the `examples` folder. This library is still in
pre-alpha and many things may break for now. The only things which will evolve at the same
pace as the library are the examples, they are meant to always be up to date with
the library.

Few examples will generates folders/files such as saved models or tensorboard logs.
To visualize the tensorboard logs download Tensorflow's tensorboard as well as 
[Pytorch's tensorboard](https://github.com/lanpa/tensorboard-pytorch). Then execute on the
log folder:
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
