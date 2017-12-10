## Neural net package

This package contains all the method/tools to deal with neural networks of all types


## Things to implement

There are several things this library lacks. Ideally it should include very
good concepts from the [fast.ai library](https://github.com/fastai/fastai) as
well as other useful libraries. A non exhaustive list of these features are 
given below:
 - [Automatic image data loader](https://github.com/fastai/fastai/blob/5200d2669d30364ac530a2e9362000c31a7cb97e/fastai/dataset.py#L264) with a `preprocessing=True` option which
 resizes and transform images to bcolz arrays
 - [Cyclical learning rate](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)
 - [Learning rate annealing](https://github.com/fastai/fastai/blob/759bfd58a15be4d7321403fd3c6e7d740f2caea9/fastai/layer_optimizer.py#L34) (To use with cyclical learning rate)
 - [TTA: Test time augmentation](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)
 - Gradient clipping
 - Automatic plot learning rates/loss
 - AdamW, SGDRW
 - Feature dependance neural net