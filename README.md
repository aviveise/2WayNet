# Linking Image and Text with 2-Way Nets
2-Wat Net is a bi-directional neural network architecture for the task of matching vectors from two data sources. The model employs two tied neural network channels that project the two views into a common, maximally correlated space optimizing the Euclidean loss. The model contain: (i) a mid-way loss term that helps support the training of the hidden layers; (ii) a decorrelation regularization term that links the problem back to CCA; (iii) modified batch normalization layers; (iv) a regularization of the scale parameter that ensures that the variance does not diminish from one layer to the next; (v) a tied dropout method; and (vi) a method for dealing with high-dimensional data.

## Installing

* Install Python 2.7 (it is recommended to use virtualenv, see https://virtualenv.pypa.io/en/stable/ for details)
* Clone this repository: ``` git clone https://github.com/aviveise/2WayNet ```
* Install prequisites: ``` pip install -r /path/to/repo/requiremens.txt ```

### Prerequisites
See the requirements.txt file for required python modules

### Running MNIST example
* Download the MNIST dataset from: http://yann.lecun.com/exdb/mnist/
* Edit MNIST.ini file and direct the dataset path to your local MNIST directory
* Modify the Params.py file to change the model's parameters
* Running training and testing execute the run_mode.py file:
``` python /path/to/repo/run_model.py /path/to/repo/MNIST.ini ```

### Credits
The 2-Way Net project was build upon the following projects:
* [Theano](http://deeplearning.net/software/theano/)
* [Lasagne](http://lasagne.readthedocs.io/en/latest/)

And other wonderful python modules