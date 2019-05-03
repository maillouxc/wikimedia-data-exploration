# Wiki Personal Attacks Data Exploration

This repository contains code and basic documentation for an independent 
study course I am taking at FGCU with Dr. Koufakou in the field of 
natural language processing.

This repository includes the actual code used to run the various 
experiments. The data is NOT included within the repository, as it
is too large for Github to allow.

However, the data can be accessed by contacting Dr. Koufakou at FGCU
or myself. It is also available on the public internet, but I am not
sure where.

The dataset used is a small subset of the Wikimedia personal attacks dataset
that I obtained from Jason Scott, another student researching at FGCU on the
same dataset with Dr. Koufakou.

Given more time, it is probably worth using the full dataset, but this could 
be a very time consuming process.

### What is here

```
results/
    - This directory will be empty initially.
      This is where the results of each experiment will ultimately be output.
data/
    - This is where you put your data
    - It also contains Jason's preprocessing script.
      This file is only needed if you intend to preprocess the data yourself.
```

### How to use these experiments?

Ensure that the data directory contains the needed data in the proper
format, then just run whatever script you wish. The results will be
output to files in the results directory of this project, depending 
which specific experiment it was. They each output accuracy, precision,
recall, and either binary f1 or macro f1, depending on the specific
script you run. They are organized into folders depending on which
type of F1 score they report. The results are output in CSV format
for easier analysis.

The data prep script in the data directory is the script supplied by
Jason Scott, another FGCU student working with Dr. Koufakou on research
on this dataset. The file is not used by any of my code but was left in
because it is needed if you preprocess the data yourself from scratch,
rather than using the already processed data that I obtained from Jason.
That file is not to be confused with the data prep file in
the root directory of the project, which is a Python file containing
functions that I have written myself to assist in data prep.

The Fasttext classifier should be very fast, and produce most results
within less than a minute on a mid-range system.

These scripts have been cleaned up to be easy to understand, tweak,
and run. All necessary dependencies are included in `requirements.txt`
and can be installed via Pip.

Additionally, performance can be improved if you have a GPU by ensuring
that you have the correct CUDA drivers properly installed, including
libcublas and cud-nn. The internet has some decent guides available
on how to do so. If these drivers are configured properly, the Tensorflow
backend should automatically take advantage of your GPU. Depending on
your specific system and how it is configured, you may already have the
necessary drivers installed.

A guide on how to do so is located at https://bit.ly/2PAMH8Q

This is probably the most difficult part of running these experiments, 
but if you follow the instructions closely it should work. Be wary of
installing too-new of a CUDA version, as in my case I had trouble with
this. The most important thing is that you have a version of
CUDA that is compatible with your GPU and with the specific version
of Tensorflow you have installed.

My system had CUDA 10.1 and NVIDIA driver 418.39, but you may need 
something different depending on your exact GPU and driver versions.

Some of these experiments take quite some time to run depending on the
specific hyperparameters chosen, but in general reducing the pad length
is the easiest way to make them run faster for testing purposes. On my
GTX 1060 6GB, all of these experiments finished in less than 48 hours,
on average giving 10 minutes per epoch on the standard-lstm/gru model.

All experiments within this repository are run with 5-fold 
cross-validation using stratified k-fold sampling.

The experiments were run on a system running Ubuntu 18.04, but should run
on most Linux distributions as long as the proper dependencies are
installed. 
