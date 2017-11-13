## High level library for Pytorch

Torchlight is a high level library meant to be what Keras is for Tensorflow and Theano.
It is not meant to micmic the Keras API at 100% but instead to get the best of both
worlds (Pytorch and Keras API). 
For instance if you used Keras train/validation generators, in Torchlight you would
use Pytorch [Dataset](http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset) and
[DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).
