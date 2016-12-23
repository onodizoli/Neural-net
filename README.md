# COMSE-6998-Project-Neural net


Queries data from a database and trains the fully connected layers of th VGG-net.  
Designed to be run on an Amazon EC2 GPU instance.

## Usage

When running training.py, it will query a traning dataset from our database. The VGG architecure is built and the fully connected layers are trained using the downloaded data. The weights of the trained network are saved for later usage. Code was written in python using theano and keras.


## Requirements

Necessary packages are included in the requirements.
