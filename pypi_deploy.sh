#!/bin/bash
rm -fR torch_light.egg-info
rm -fR dist
rm -fR build
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
