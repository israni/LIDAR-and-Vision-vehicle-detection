#! /usr/bin/python3
# Sorts labeled training images into folders in the format tensorflow retraining suggest

import os
from shutil import copyfile
import IPython

LOAD_DIRECTORY = os.path.join('..','labeled_training_data')
SAVE_DIRECTORY = os.path.join('..','tf_retraining_data')


labels_file = os.path.join(LOAD_DIRECTORY, 'labels.txt')

with open(labels_file) as labels:
    next(labels) #Skip header line
    for line in labels:
        fp, label = line.split(',')
        label = label.strip('\n')
        parts = fp.split(os.sep)

        outdir = os.path.join(SAVE_DIRECTORY, label)
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, parts[-3] + '_' + parts[-2] + '_' + parts[-1])

        copyfile(fp, save_path)

