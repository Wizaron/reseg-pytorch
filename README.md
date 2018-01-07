# Implementation of ReSeg using PyTorch

* [ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation](https://arxiv.org/abs/1511.07053)
* [Pascal-Part Annotations](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html)
* [Pascal VOC 2010 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit)
# Setup

* Clone this repository : `git clone --recursive https://github.com/Wizaron/reseg-pytorch.git`
* Download [Pascal-Part Annotations](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html) and [Pascal VOC 2010 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit) to "reseg-pytorch/data/raw" then extract tar files.
* Go to the "reseg-pytorch/code/pytorch" : `cd reseg-pytorch/code/pytorch`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Create environment : `conda env create -f pytorch_conda_environment.yml`
