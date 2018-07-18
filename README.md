# DeepGLM
Version 0.1.0<br/>

## Introduction
DeepGLM is a flexible model that use Deep Feedforward Neuron Network as the basis function for Generalized Linear Model. DeepGLM is designed to work with Cross-Sectional Dataset such as real estate data, cencus data, etc. <br/>

For more information about DeepGLM, please read the paper: Minh-Ngoc Tran,Nghia Nguyen, David J. Nott and Robert Kohn (2018)  Bayesian Deep Net GLM and GLMM https://arxiv.org/abs/1805.10157

## Authors
Nghia Nguyen (nghia.nguyen@sydney.edu.au) <br/>
Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)

## Usage
Users can choose either Matlab, R or Python version to train and make prediction with deepGLM.
### MATLAB Version
To use the Toolbox, add the folder called "deepGLM" (with Subfolders) to the MATLAB path.

The toolbox contains the following folders:

- Data: some datasets used in the examples.
- Examples: examples of all the functions included in the toolbox.
- Documents: documentations for the functions in deepGLM toolbox
- deepGLM: all the functions of the toolbox all here. This is the folder you must add to the MATLAB path.

### R Version
Install *deepglm* package for R:
- Clone the directory or directly download the zip file **deepglm_0.0.0.9000.zip** inside *deepGLM/R/* subdirectory on github. 
- In Rstudio, run the command: 
**install.packages("D:\\deepglm_0.0.0.9000.zip", repos = NULL, type="source")**
where *D:\deepglm_0.0.0.9000.zip* is the package directory in local machine

Use *deepglm* package:
- *deepglm* provides two function to train a deepGLM model on training data (***deepGLMfit***) and to make prediction using a trained deepGLM model on unseen data (***deepGLMpredict***)

### Python Version
Coming soon...

## How to cite
Please, cite the toolbox as:

Tran, M.-N., Nguyen, N., Kohn, R., and Nott, D. (2018) Bayesian Deep Net GLM and GLMM. *arXiv preprint arXiv:1805.10157*
