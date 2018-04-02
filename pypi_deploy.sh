#!/bin/bash
rm -fR torchlite.egg-info
rm -fR dist
rm -fR build
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
