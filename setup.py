from setuptools import setup, find_packages
import torchlite

with open('README.md') as f:
    long_description = f.read()

setup(
    name='torchlite',
    version=torchlite.__version__,
    description='A high level library on top of Pytorch',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],

    keywords='development',
    packages=find_packages(exclude=['tests']),
    install_requires=["isoweek", "tqdm", "opencv_python", "torch", "torchvision",
                      "scikit_image", "setuptools", "numpy", "pandas", "matplotlib", "scipy",
                      "Pillow", "scikit_learn", "tensorboardX", "PyYAML", "Augmentor",
                      "fuzzywuzzy", "python-Levenshtein", "category_encoders",
                      find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                             "tests", "torchlite.*", "torchvision.*"])],
)
