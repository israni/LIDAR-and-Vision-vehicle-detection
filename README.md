# Perception

## Setup on linux (tested on Ubuntu 16.04)
Install python3 and tensorflow (preferably in a virtualenv)

    cd ~
    virtualenv -p python3 tf3
    source ~/tf3/bin/activate
    sudo apt install python3-tk
    pip install tensorflow matplotlib pillow scikit-learn imageio networkx IPython scipy


Download the [training data](http://umich.edu/~fcav/rob599_dataset_deploy.zip) and place in a folder at the same level as the Perception folder (not in this git repo - that would be too large)
Run `python ./count_test` to start computing labels for the test data. This will write to outfile.txt.
To view this file in real time: `tail -f outfile.txt`

To play around with the training data, run `python ./tmp`. Using this file for testing random changes.

To retrain the tensorflow network:
- Look at [Tensorflow for poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/)
- generate training images using `python generate_labeled_training_data.py` (will take several hours)
- put the data in correct format using `python organize_labeled_data.py`



## Overview
The basic algorithms is:
1. Filter lidar data to remove ground points and uninteresting data
2. Segment lidar data into clusters of interest
3. Get corresponding images of interest
4. use tensorflow to classify these images of interest, counting the number of cars
    
    