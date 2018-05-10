from setuptools import setup, find_packages
import torchlite

with open('README.md') as f:
    long_description = f.read()

setup(
    name='torchlite',
    version=torchlite.__version__,

    description='A high level library on top of machine learning frameworks',
    long_description=long_description,
    # Markdown on pypi https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
    long_description_content_type="text/markdown",
    url='https://github.com/EKami/Torchlite',
    author='GODARD Tuatini',
    author_email='tuatinigodard@gmail.com',
    license='MIT',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    keywords='development',
    packages=find_packages(exclude=['tests']),
    install_requires=["isoweek", "tqdm", "bcolz", "kaggle_data", "opencv_python", "torch", "torchvision",
                      "tensorflow-gpu", "scikit_image", "setuptools", "numpy", "matplotlib", "scipy", "Pillow",
                      "scikit_learn", "tensorboardX", "typing", "PyYAML", "Augmentor", "feather-format",
                      "fuzzywuzzy", "python-Levenshtein", "category_encoders",
                      find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                             "tests", "torchlite.*", "torchvision.*"])],
)
