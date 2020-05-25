import numpy as np
from scipy.io import loadmat
import numbers
from scipy.special import gammaln
from scipy.special import psi
from numpy.linalg import det, norm
from numpy import log, sum, power, outer, dot, sqrt, arange, diag, \
                    concatenate, ones, zeros, mean, argsort, std
from numpy.random import permutation, multivariate_normal, gamma, normal
import time

import matplotlib.pyplot as plt
import matplotlib

class deepGLMout():
# DEEPGLMOUT Generate default output structure for deepGLM training results

#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    def __init__(self,
        dist = 'normal',
        initialize = 'adaptive',
        isIsotropic = False,
        ncore = 0,
        seed = np.nan,
        nval = 0.2,
        verbose = 10,
        cutoff = 0.5,
        stop = False,
        quasiMC = True,
        monitor = False,
        muTau = np.nan,
        lowerbound = True,
        windowSize = 100,
        network = [10,10],
        lrate = 0.01,
        S = 10,
        batchsize = 5000,
        epoch = 1000,
        tau = 10000,
        patience = 100,
        c = 0.01,
        bvar = 0.01,
        momentum = 0.6,
        ynames = np.nan,
        xnames = np.nan,
        y = np.nan,
        X = np.nan,
        ytest = np.nan,
        Xtest = np.nan,
        nTrain = np.nan,
        nTest = np.nan,
        Xval = [],
        yval = [],
        icept = True):
        
        # Training method
        self.dist = dist         # Default distribution of dependent variable. 
        self.initialize = initialize # Default initialize method
        self.isIsotropic = isIsotropic     # Default structure of variational Covariance matrix
        self.ncore = ncore               # Default parallel computing option

        # Optional settings
        self.seed = seed              # No random seed by default
        self.nval = nval              # Default proportion of training data for validation
        self.verbose = verbose            # Default number of iteration to display training results
        self.cutoff = cutoff            # Default Cutoff probability for sigmoid function
        self.stop = stop            # Execution Flag
        self.quasiMC = quasiMC          # Using Quasi MC for random number generator
        self.monitor = monitor         # Display training progress window
        self.muTau = muTau
        self.lowerbound = lowerbound
        self.windowSize = windowSize

        # Model hyper-parameters
        self.network = network    # Default network structure
        self.lrate = lrate           # Default Learning rate
        self.S = S                  # Default Number of samples used to approximate gradient of likelihood
        self.batchsize = batchsize         # Default Proportion of batch size over entire train set
        self.epoch = epoch             # Default Number of epoches in train phase
        self.tau = tau             # Default Scale factor of learning rate
        self.patience = patience           # Default Number of consequence non-decreasing iterations (for early stopping checking)
        self.c = c                # Default initial value of isotropic factor c
        self.bvar = bvar             # Default initial variance of each element of b
        self.momentum = momentum          # Default momentum weight

        # Variable names
        class deepGLMout_name():
            def __init__(self, ynames=ynames, xnames=xnames):
                self.ynames = ynames      # y variables names
                self.xnames = xnames      # X variable names

        self.name = deepGLMout_name()

        # Data properties
        class deepGLMout_data():
            def __init__(self,
                         y = y,
                         X = X,
                         ytest = ytest,
                         Xtest = Xtest,
                         nTrain = nTrain,
                         nTest = nTest,
                         Xval = Xval,
                         yval = yval,
                         icept = icept
                        ):
                self.y = y            # Dependent variable of training data
                self.X = X            # Independent variables of training data
                self.ytest = ytest        # Dependent variable of test data
                self.Xtest = Xtest        # Independent variables of tets data
                self.nTrain = nTrain       # Number of observation in training set
                self.nTest = nTest        # Number of observation in test set
                self.Xval = Xval
                self.yval = yval
                self.icept = icept      # Intercept option
        self.data = deepGLMout_data()

        # Training results
        class deepGLMout_out():
            def __init__(self, mse=np.nan, accuracy=np.nan):
                self.mse = mse
                self.accuracy = accuracy
        self.out = deepGLMout_out()
        
def deepGLMmsg(identifier):
#DEEPGLMMSG Define custom error/warning messages for exceptions
#   DEEPGLMMSG = (IDENTIFIER) extract message for input indentifier
#   
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    if identifier == 'deepglm:TooFewInputs':
        return 'At least two arguments are specified'
    elif identifier == 'deepglm:InputSizeMismatchX':
        return 'X and Y must have the same number of observations'
    elif identifier == 'deepglm:InputSizeMismatchY':
        return 'Y must be a single column vector'
    elif identifier == 'deepglm:ArgumentMustBePair':
        return 'Optinal arguments must be pairs'
    elif identifier == 'deepglm:ResponseMustBeBinary':
        return 'Two level categorical variable required'
    elif identifier == 'deepglm:DistributionMustBeBinomial':
        return 'Binomial distribution option required'
    elif identifier == 'deepglm:MustSpecifyActivationFunction':
        return 'Activation function type required'
    
# seeded rand
def my_rand(a,b):
    return np.random.rand(b,a).transpose()

def my_randn(a,b):
    return np.random.randn(b,a).transpose()

def my_reshape(x,a,b):
    return np.reshape(x.transpose(), [b,a]).transpose()
    
def nnInitialize(layers):
#NNINITIALIZE Summary of this function goes here
#  layers: vector of doubles, each number specifing the amount of
#  nodes in a layer of the network.
#
#  weights: cell array of weight matrices specifing the
#  translation from one layer of the network to the next.
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    weights = [[] for i in range(len(layers)-1)]

    for i in range(len(layers)-1):
        # Using random weights from -b to b 
        b = np.sqrt(6)/(layers[i]+layers[i+1]);
        if i==0:
            weights[i] = my_rand(layers[i+1],layers[i])*2*b - b  # Input layer already have bias
        else:
            weights[i] = my_rand(layers[i+1],layers[i]+1)*2*b - b  # 1 bias in input layer
    
    return weights

def nnActivation(z,func):
#NNACTIVATION Calculate activation output at nodes in each forward pass

#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    if func == 'Linear':
        out = z
    elif func == 'Sigmoid':
        out = 1.0 / (1.0 + np.exp(-z))
    elif func == 'Tanh':
        out = np.tanh(z)
    elif func == 'ReLU':
        out = np.maximum(z, 0)
    elif func == 'LeakyReLU':
        out = np.maximum(0,z)+ alpha*np.min(0,z)
    else:
        raise('activation function must be either Linear, Sigmoid, Tanh, RelU, LeakyReLU')
        
    return out

def nnFeedForward(X,W_seq,beta):
#NNFEEDFORWARD Compute the output of a neural net 

#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    # Number of observations in dataset
    n_train = X.shape[0] 

    # Make forward passes to all layers
    a = np.dot(W_seq[0],X.transpose())
    Z = np.concatenate([np.ones([1,n_train]), nnActivation(a,'ReLU')], axis=0)
    L = len(W_seq)
    for j in range(1,L):
        a = np.dot(W_seq[j],Z)
        Z = np.concatenate([np.ones([1,n_train]), nnActivation(a,'ReLU')], axis=0) # Add biases 
        
    # a = W_seq{L}*Z;
    # Z = [ones(1,n_train);nnActivation(a,'ReLU')];
    nnOutput = np.dot(Z.transpose(),beta)

    return nnOutput

def deepGLMpredictLoss(X,y,W_seq,beta,distr,sigma2):
#DEEPGLMPREDICTION Make prediction from estimated deepGLM model
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    # Calculate neuron network output
    nnet_output = nnFeedForward(X,W_seq,beta)

    if distr == 'normal':
        mse = np.mean(np.power(y-nnet_output, 2))
        pps = 1/2*np.log(sigma2) + 1/2/sigma2*mse
        out2 = mse
    elif distr == 'binomial':
        pps = np.mean(-np.multiply(y,nnet_output) + np.log(1+np.exp(nnet_output)))
        y_pred = nnet_output>0
        mcr = np.mean(np.abs(y-y_pred))        # Miss-classification rate
        out2 = 1 - mcr                   # Report output in classification rate   
    elif distr == 'poisson':
        pps = np.mean(-p.multiply(y, nnet_output) + np.exp(nnet_output))
        mse = np.mean(np.power(y-np.exp(nnet_output),2))
        out2 = mse
        
    out1 = pps

    return out1, out2

def toVector(matrix):
    temp = matrix.flatten()
    return temp.reshape([len(temp),1])

def nnBackPropagation(X,y,W_seq,beta,distr):
#NNBACKPROPAGATION Compute gradient of weights in a neural net using 
# backpropagation algorithm
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    n_train = X.shape[0]
    L = len(W_seq)
    a_seq = [[] for i in range(L)]
    Z_seq = [[] for i in range(L)]

    a_seq[0] = dot(W_seq[0], X.transpose())
    
    Z_seq[0] = concatenate([ones([1,n_train]),nnActivation(a_seq[0],'ReLU')], axis=0)
    for j in range(1,L):
        a_seq[j] = dot(W_seq[j], Z_seq[j-1])
        Z_seq[j] = concatenate([ones([1,n_train]),nnActivation(a_seq[j],'ReLU')], axis=0)
        
    delta_seq = [[] for i in range(L+1)]

    # Calculate error at the output layers according to distribution family of
    # response
    nnOut = dot(beta.transpose(), Z_seq[L-1])
    if distr == 'normal':
        delta_seq[L] = y.transpose() - nnOut
    elif distr == 'binomial':
        p_i = 1/(1+exp(-nnOut))
        delta_seq[L] = y.transpose() - p_i
    elif distr == 'poisson':
        delta_seq[L] = y.transpose() - exp(nnOut)
    
    delta_seq[L-1] = dot(beta[1:],delta_seq[L])*nnActivationGrad(a_seq[L-1],'ReLU')
    for j in range(L-2, -1, -1):
        Wj_tilde = W_seq[j+1]
        Wj_tilde = Wj_tilde[:,1:]
        delta_seq[j] = (nnActivationGrad(a_seq[j],'ReLU'))*dot(Wj_tilde.transpose(),delta_seq[j+1])
        
    gradient_W1 = dot(delta_seq[0],X).transpose()
    gradient = toVector(gradient_W1)
    for j in range(1,L):
        gradient_Wj = dot(delta_seq[j],(Z_seq[j-1]).transpose()).transpose()
        gradient = concatenate([gradient,toVector(gradient_Wj)], axis=0)
    
    gradient = concatenate([gradient,dot(Z_seq[L-1],delta_seq[L].transpose())])
    
    return gradient, nnOut

def nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse):
#NNGRADIENTLLH Calculate gradient of log likelihood
#   Detailed explanation goes here
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    n = len(y)
    [back_prop,nnOut] = nnBackPropagation(X,y,W_seq,beta,distr);
    nnOut = nnOut.transpose()
    if distr == 'normal':
        gradient_theta = dot(mean_sigma2_inverse, back_prop)
        gradient = datasize/n*gradient_theta   # To compensate the variation
    elif distr == 'binomial':
        gradient = datasize/n*back_prop
    elif distr == 'poisson':
        gradient = datasize/n*back_prop

    return gradient, nnOut

def vbGradientLogq(b,c,theta,mu,isotropic):
#VBGRADIENTLOGQ Summary of this function goes here
#   Detailed explanation goes here
    x = theta-mu
    if isotropic:
        grad_log_q = -x/power(c,2)+(1/power(c,2))\
                        *dot((dot(b.transpose(),x)/(power(c,2)+dot(b.transpose(),b))),b)
    else:
        d = b/power(c,2)
        grad_log_q = -x/power(c,2) + (dot(d.transpose(),x)/(1+dot(d.transpose(),b)))*d
        
    return grad_log_q

def nnActivationGrad(z,func):
#NNACTIVATIONGRAD Calculate derivative of activation output at hidden nodes 
#in each backward pass
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    if func == 'Linear':
        out = ones(z.shape)
    elif func == 'Sigmoid':
        temp = activation(z,text)
        out = temp*(1-temp)
    elif func == 'Tanh':
        temp = activation(z,text)
        out = 1 - power(temp,2)
    elif func == 'ReLU':
        out = z>0
    elif 'LeakyReLU':
        if z > 0:
            out = 1
        else:
            out = alpha
            
    return out


def vbNaturalGradient(b,c,grad,isotropic):
#VBNATURALGRADIENT compute the product inverse_fisher times grad for two 
# cases: isotropic factor decompostion or rank-1 decomposition
# INPUT: 
#   grad:           the traditional gradient
#   b,c:            parameters in the factor decomposition
#   isotropic:      true if isotropic structure is used, rand-1 otherwise
# 
# OUTPUT: natural gradient
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au).
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    if isotropic:
        d = len(b)
        bb = dot(b.transpose(),b)
        alpha = 1/(c**2+bb);
        omega = (2/c**2)*(d-1+(c**4)*alpha**2);
        kappa = (1+(c**2)/bb -.5*(1+(c**2)/bb)**2)*2*c*bb*alpha**2 + 2*(c**3)*alpha/bb;
        c2 = omega-2*c*(alpha**2)*kappa*bb;

        grad1 = grad[:d]
        grad2 = grad[d:2*d]
        grad3 = grad[-1]

        b_grad2 = dot(b.tranpose(),grad2)
        const1 = (1 + (c**2)/bb - .5*(1 + c**2/bb)**2)
        const2 = c**2*(1 + (c**2)/bb)
        Ainv_times_grad2 = (const1*b_grad2)*b + const2*grad2

        prod = concatenate([dot(b.transpose(),grad1)*b + (c**2)*grad1,
                            Ainv_times_grad2 + (kappa**2/c2*b_grad2)*b - (kappa/c2*grad3)*b,
                            -kappa/c2*b_grad2 + grad3/c2], axis=1)
    else:
        d = len(b)
        grad1 = grad[:d]
        grad2 = grad[d:2*d]
        grad3 = grad[2*d:]

        c2 = power(c,2)
        b2 = power(b,2)

        prod1 = dot(b.transpose(),grad1)*b + np.multiply(grad1, c2)

        const = np.sum(b2/c2)
        const1 = 1/2 + 1/2/const
        prod2 = dot(b.transpose(),grad2)*b + (grad2*c2)
        prod2 = const1*prod2
        alpha = 1/(1+const)
        x = alpha*b2/power(c,3)
        y = 1/c2 - 2*alpha*power(b/c2,2)
        aux = x/y    
        prod3 = grad3/y - (1/(1+np.sum(power(x,2)/y)))*dot(aux.transpose(),grad3)*aux
        prod3 = prod3/2;
        prod = concatenate([prod1, prod2, prod3], axis=0)

    return prod

def sumResidualSquared(y,X,W_seq,beta):
    # compute the sum_residual_squared for normal-NN model

    nnet_output = nnFeedForward(X,W_seq,beta)
    S = np.sum(power(y-nnet_output, 2))
    return S

def checkInput(est):
#CHECKDATA Check if user input correct model settings
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au).
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018
    dist = est.dist
    network = est.network
    lrate = est.lrate
    momentum = est.momentum
    batchsize = est.batchsize
    epoch = est.epoch
    patience = est.patience
    tau = est.tau
    S = est.S
    windowSize = est.windowSize
    icept = est.icept
    verbose = est.verbose
    monitor = est.monitor
    isotropic = est.isIsotropic
    seed = est.seed
    
    # function checkInput(est)
    if dist not in ['normal','binomial', 'poisson']:
        raise ValueError('Distribution must be one of the followings: ','normal,','binomial,','poisson')

    if (sum((np.array(network)==0))>0):
        raise ValueError('Network must be an array of positive integers')

    isnumeric = lambda x: isinstance(x, numbers.Number)
    if (not isnumeric(lrate) or (lrate<=0)):
        raise ValueError('Learning rate must be a positive numerical value')

    if (not isnumeric(momentum) or (momentum<0) or (momentum > 1)):
        raise ValueError('Momentum must be a numerical value from 0 to 1')

    if (not isnumeric(batchsize) or (np.floor(batchsize)!= batchsize) or (batchsize <= 0)):
        raise ValueError('Batch size must be an positive integer smaller than number of observations in training data')

    if (not isnumeric(epoch) or (np.floor(epoch)!= epoch) or (epoch <= 0)):
        raise ValueError('Number of epoches must be a positive integer')

    if (not isnumeric(patience) or (np.floor(patience)!= patience) or (patience <= 0)):
        raise ValueError('Patience must be a positive integer')

    if (not isnumeric(tau) or (np.floor(tau)!= tau) or (tau <= 0)):
        raise ValueError('LrateFactor must be a positive integer')

    if (not isnumeric(S) or (np.floor(S)!= S) or (S <= 0)):
        raise ValueError('S must be a positive integer')

    if (not isnumeric(windowSize) or (np.floor(windowSize)!= windowSize) or (windowSize <= 0)):
        raise ValueError('WindowSize must be a positive integer')

    islogical = lambda x: isinstance(x, type(True))
    if (not islogical(icept)):
        raise ValueError('Intercept option must be a logical value')

    if (not isnumeric(verbose) or (np.floor(verbose)!= verbose) or (verbose <= 0)):
        raise ValueError('Verbose must be a positive integer')

    if(not islogical(monitor)):
        raise ValueError('Monitor option must be a logical value')

    if(not islogical(isotropic)):
        raise ValueError('Isotropic option must be a logical value')

    if (not np.isnan(seed)):
        if (not isnumeric(seed) or (np.floor(seed)!= seed) or (seed <= 0)):
            raise ValueError('Seed must be a nonnegative integer less than 2^32')
            
def deepGLMfit(X,y,
        dist = 'normal',
        initialize = 'adaptive',
        isotropic = False,
        ncore = 0,
        seed = np.nan,
        nval = 0.2,
        verbose = 10,
        cutoff = 0.5,
        quasiMC = True,
        monitor = False,
        muTau = np.nan,
        lowerbound = True,
        windowSize = 100,
        network = [10,10],
        lrate = 0.01,
        S = 10,
        batchsize = 5000,
        epoch = 1000,
        tau = 10000,
        patience = 100,
        c = 0.01,
        bvar = 0.01,
        momentum = 0.6,
        Xval = [],
        yval = [],
        icept = True,
        ):
#DEEPGLM Traing a deepGLM model. DeepGLM is a flexible version of Generalized 
# Liner Model where Deep Feedforward Network is used to automatically choose 
# transformations for the raw covariates. Bayesian Adaptive Group Lasso is 
# used on the first-layer weights; a Ridge-type prior is imposed to the rests. 
# sigma2 and tau are updated by mean-field VB. Inverse gamma prior is used for 
# sigma2
# 
#   MDL = DEEPGLMFIT(X,Y) fits a deepGLM model using the design matrix X and 
#   response vector Y, and returns an output structure mdl to make prediction 
#   on a test data. By default, if 'distribution' option is not specified, 
#   deepGLMfit will treat response variable y as normal distributed variable.
#
#   MDL = DEEPGLMFIT(X,Y,NAME,VALUE) fit a deepGLM model with additional options
#   specified by one or more of the following NAME/VALUE pairs:
#
#      'Distribution'     Name of the distribution of the response, chosen
#                         from the following:
#                 'normal'             Normal distribution (default)
#                 'binomial'           Binomial distribution
#                 'poisson'            Poisson distribution
#      'Network'          Deep FeedforwardNeuron Network structure for deepGLM. 
#                         In the current version, deepGLM supports only 1 node 
#                         for the output layer, users just need to provide a 
#                         structure for hidden layers in an array where each 
#                         element in the array is the 
#                         number of nodes in the corresponding hidden layer.
#      'Lrate'            Vector of integer or logical indices specifying
#                         the variables in TBL or the columns in X that
#                         should be treated as categorical. Default is to
#                         treat TBL variables as categorical if they are
#                         categorical, logical, or char arrays, or cell
#                         arrays of strings.
#      'Momentum'         Momentum weight for stochastic gradient ascend. 
#                         The momentum determines the contribution of the 
#                         gradient step from the previous iteration to the 
#                         current iteration of training. It must be a value 
#                         between 0 and 1, where 0 will give no contribution 
#                         from the previous step, and 1 will give a maximal 
#                         contribution from the previous step. Must be between 
#                         0 and 1. 
#      'BatchSize'        The size of the mini-batch used for each training 
#                         iteration. Must be a positive integer smaller than 
#                         number of observations of training data
#      'MaxEpoch'         The maximum number of epochs that will be used for 
#                         training. An epoch is defined as the number of 
#                         iterations needed for optimization algorithm to 
#                         scan entire training data. Must be a positive integer.
#      'Patience'         Number of consecutive times that the validation loss 
#                         is allowed to be larger than or equal to the previously 
#                         smallest loss before network training is stopped, 
#                         used as an early stopping criterion. Must be a positive 
#                         integer.
#      'LrateFactor'      Down-scaling factor that is applied to the learning 
#                         rate every time a certain number of iterations has 
#                         passed. Must be a positive integer
#      'S'                The number of samples needed for Monte Carlo 
#                         approximation of gradient of lower bound. Must 
#                         be an positive integer
#      'WindowSize'       Size of moving average window that used to smooth 
#                         the VB lowerbound. Must be an positive integer
#      'Intercept'        Set true (default) to add a column of 1 to predictor 
#                         observation X matrix (play the role as intercept). 
#                         If the data have already included the first '1' column, 
#                         set 'Intercept' to false.
#      'Verbose'          Number of iterations that information on training 
#                         progress will be printed to the command window each 
#                         time. Set to 0 to disable this options. 
#      'Monitor�'         Display monitor window showing the training process 
#                         on a user interface. This is a useful tool to visualize 
#                         training metrics at every iteration. However, using 
#                         this option will slow down training progress because 
#                         of graphical related tasks.
#      'Isotropic'        Set to true if you want to use Isotropic structure 
#                         on Sigma (Variational Covariance matrix). By default, 
#                         deepGLM uses rank-1 structure to factorize Sigma
#      'Seed'             Seeds the random number generator using the nonnegative 
#                         integer. Must be a nonnegative integer.
#
#   Example:
#      Fit a deepGLM model for Direcmarketing data set. All of the
#      exampled data are located inside /Data folder of installed package. 
#      In order to use the sample dataset, user must add this Data folder
#      to Matlab path or explicitly direct to Data folder in 'load'
#      function
#
#      load('DirectMarketing.mat')
#      mdl = deepGLMfit(X,y,...                   % Training data
#                      'Network',[5,5],...        % Use 2 hidden layers
#                      'Lrate',0.01,...           % Specify learning rate
#                      'Verbose',10,...           % Display training result each 10 iteration
#                      'BatchSize',size(X,1),...  % Use entire training data as mini-batch
#                      'MaxEpoch',10000,...       % Maximum number of epoch
#                      'Patience',50,...          % Higher patience values could lead to overfitting
#                      'Seed',100);               % Set random seed to 100
#
#   For more examples, check EXAMPLES folder
#
#   See also DEEPGLMPREDICT, DEEPGLMPLOT
#
#   Copyright 2018:
#       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
#       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
#      
#   https://github.com/VBayesLab/deepGLM
#
#   Version: 1.0
#   LAST UPDATE: May, 2018

    # Initialize output structure with default setting
    est = deepGLMout();

    # Check errors input arguments
    if y.shape[0] != X.shape[0]:
        raise ValueError(deepGLMmsg('deepglm:InputSizeMismatchX'))

    if y.shape[1] != 1:
        raise ValueError(deepGLMmsg('deepglm:InputSizeMismatchY'))

    if any(np.isnan(y)) or any(np.isnan(X.flatten())): # Check if data include NaN
        raise ValueError('NaN values not allowed in input data')
        
    # Check errors for additional options
    # If distribution is 'binomial' but responses are not binary/logical value
    isBinomial = lambda x: not any(\
        np.array(list(map(lambda y: y not in [0, 1, 0.0, 1.0, True, False], x.flatten()))))
    if (isBinomial(y)!=True) and (dist=='binomial'):
        raise ValueError(deepGLMmsg('deepglm:ResponseMustBeBinary'))

    # If response is binary array but distribution option is not 'binomial'
    if (isBinomial(y) and (dist!='binomial')):
        raise ValueError(deepGLMmsg('deepglm:DistributionMustBeBinomial'))

    ## Prepare data and parameters for training phase
    # If lowerbound option is true -> do not need validation data
    if(lowerbound):
        Xval = []
        yval = []
    else:
        n = X.shape[0]                # Number of observation in input dataset
        if (nval <= 1):
            numVal = round(nval*n)       # Number of observation in validation set
        else:
            numVal = nval

        numTrain = n - numVal        # Number of observation in training
        # Extract validation set from training data if validation data are not specified
        if (not np.isnan(seed)): # Set random seed if specified
            np.random.seed(seed)

        isempty = lambda x: len(x.flatten()) == 0
        if (isempty(Xval) and isempty(yval)):
            idxVal = np.random.permutation(range(n))[:numVal]     # Random indexes of validation data
            Xval = X[idxVal,:]              # Extract subset from input data for validation
            yval = y[idxVal]
            X[idxVal,:] = []                # Training data
            y[idxVal]   = [] 

    if (icept):                        # If intercept option is turned on
        X = np.concatenate([np.ones([X.shape[0],1]), X], axis=1)     # Add column of 1 to data

    ## Calculate batchsize in stochastic gradient ascend
    if (batchsize <= 1):              # If specified batchsize is a propotion
        batchsize = batchsize * X.shape[0]

    if (batchsize >= X.shape[0]):
        batchsize = X.shape[0]

    if (batchsize > X.shape[0]):
        raise ValueError('Batch size must be an positive integer smaller than number of observations in training data')

    ## Store training settings
    est.isIsotropic = isotropic;
    est.S = S;
    est.batchsize = batchsize;
    est.lrate = lrate;
    est.initialize = initialize;
    est.ncore = ncore;
    est.epoch = epoch;
    est.tau = tau;
    est.patience = patience;
    est.network = np.floor(network);
    est.dist = dist;
    est.seed = seed;
    est.c = c;
    est.cutoff = cutoff;
    est.bvar = bvar;
    est.nval = nval;
    est.icept = icept;
    est.momentum = momentum;
    est.verbose = verbose;
    est.quasiMC = quasiMC;
    est.muTau = muTau;
    est.lowerbound = lowerbound;
    est.windowSize = windowSize;
    est.monitor = monitor;
    est.data.Xval = Xval;
    est.data.yval = yval;   
    
    # Check if inputs are valid
    checkInput(est);
    
    ## Run different models based on different types of distribution families
    if monitor:          # If user want to look at training progress
        est = DeepGLMTrainMonitor(X,y,est);
    else:                # Run training using Matlab scripts
        tic = time.time()
        est = deepGLMTrain(X,y,est);
        CPU = time.time() - tic
        print('Training time: %ss' % CPU)
        est.out.CPU = CPU      # Save training time
        
    return est

def vbLowerBound(beta_sigma2, alpha_sigma2,
                b, c, distr, p,
                alpha0_sigma2, beta0_sigma2,
                mean_sigma2_inverse, n_units, 
                shrinkage_gamma, mean_tau,
                datasize, lambda_tau,
                d_w_tilde, shrinkage_l2,
                mean_w_tilde, mean_column_j_tilde,
                mean_inverse_tau):

    #  vbLowerBound
    ## Group Lasso + L2 prior on remaining weigths
    if distr == 'normal':
        mean_log_sig2 = np.log(beta_sigma2) - psi(alpha_sigma2)
        logdet = log(det(1 + dot((b/power(c,2)).transpose(),b))) + sum(log(power(c,2)))   
        constMean = -(alpha0_sigma2+1)*mean_log_sig2 - beta0_sigma2*mean_sigma2_inverse
        constMean += +0.5*sum(2*(n_units[0]+1)*log(shrinkage_gamma)- power(shrinkage_gamma,2)*mean_tau)
        constMean += -0.5*datasize*mean_log_sig2+gammaln(alpha_sigma2)
        constMean += -alpha_sigma2*log(beta_sigma2)+(alpha_sigma2+1)*mean_log_sig2
        constMean += +alpha_sigma2-0.5*(sum(log(lambda_tau))-p)+0.5*logdet
        constMean += +0.5*d_w_tilde*log(shrinkage_l2)-0.5*shrinkage_l2*mean_w_tilde
        constMean += -0.5*sum(dot(mean_column_j_tilde.transpose(),mean_inverse_tau))
    else:
        logdet = log(det(1 + dot((b/power(c,2)).transpose(),b))) + sum(log(power(c,2)));   
        constMean = 0.5*sum(2*(n_units[0]+1)*log(shrinkage_gamma)-power(shrinkage_gamma,2)*mean_tau)
        constMean += -0.5*(sum(log(lambda_tau))-p)+0.5*logdet+0.5*d_w_tilde*log(shrinkage_l2)
        constMean += -0.5*shrinkage_l2*mean_w_tilde-0.5*sum(dot(mean_column_j_tilde.transpose(),mean_inverse_tau))

    return logdet, constMean

def vbGradientLogLB(S, d_theta, mu, L, index_track, n_units, p,
                   beta, X, y, datasize, distr, mean_sigma2_inverse,
                   b, c, isotropic, mean_inverse_tau, shrinkage_l2,
                   d_w, lbFlag, constMean, batchsize, const,
                   grad_g_lik_store):

    ## Calcualte for the first iteration
    lb_iter = zeros(S);

    ########################vbGradientLogLB######################################
    #----------------------------Narutal Gradient (1st Iteration)--------------

    ## Script to calculate natural gradient of lowerbound
    #rqmc = normrnd_qmc(S,d_theta+1);     % Using quasi MC random numbers 
#     np.random.seed(1)
    rqmc = my_randn(S,d_theta+1)
#     rqmc = np.arange(S*(d_theta+1)).reshape([S, d_theta+1])/100 ### Test random ###############################
    
    # rng(iter)
    # rqmc = rand(S,d_theta+1);
    for s in range(S):
        U_normal = rqmc[s,:].reshape([1, d_theta+1]).transpose()
        epsilon1 = toVector(U_normal[0])
        epsilon2 = toVector(U_normal[1:])
        theta=mu+epsilon1*b+c*epsilon2   

        W_seq = [[] for i in range(L)]       
        W1 = my_reshape(theta[arange(index_track[0])], n_units[0], p+1)
        W_seq[0] = W1
        W1_tilde = W1[:,1:] # weights without biases
        W1_tilde_gamma = dot(W1_tilde, diag(mean_inverse_tau.flatten()))
        grad_prior_w_beta = concatenate([zeros([n_units[0],1]), toVector(-W1_tilde_gamma.transpose())], axis=0) 
        for j in range(1,L):
            index = arange(index_track[j-1],index_track[j])
            Wj = my_reshape(theta[index], n_units[j], n_units[j-1]+1)
            W_seq[j] = Wj 
            Wj_tilde = Wj[:,1:]
            grad_prior_Wj = concatenate([zeros([n_units[j],1]),toVector(-shrinkage_l2*Wj_tilde.transpose())], axis=0)        
            grad_prior_w_beta = concatenate([grad_prior_w_beta,grad_prior_Wj], axis=0)
        
        beta = theta[arange(d_w,d_theta)]
        beta_tilde = beta[1:] # vector beta without intercept
        grad_prior_beta = concatenate([zeros([1,1]),-shrinkage_l2*beta_tilde], axis=0)
        grad_prior_w_beta = concatenate([grad_prior_w_beta,grad_prior_beta], axis=0)

        if (distr == 'normal'):    
            grad_llh,yNN = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse)
        else:
            grad_llh,yNN = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr)
        
        grad_h = grad_prior_w_beta+grad_llh    # Gradient of log prior plus log-likelihood
        grad_log_q = vbGradientLogq(b,c,theta,mu,isotropic)
        grad_theta = grad_h-grad_log_q
        temp = concatenate([grad_theta,
                             epsilon1*grad_theta,
                             epsilon2*grad_theta], axis=0)
        grad_g_lik_store[s,:] = temp.transpose()

    #   ------------------ lower bound ---------------------------------------
        if (lbFlag):
            if distr == 'normal':
                lb_iter[s] = constMean\
                            -0.5*mean_sigma2_inverse*sum(power(y-yNN,2))*datasize/batchsize\
                            +const
            elif distr == 'binomial':
                lb_iter[s] = constMean\
                            +sum(y*yNN - log(1+exp(yNN)))*datasize/batchsize\
                            +const
            else:
                lb_iter[s] = constMean\
                            +sum(y*yNN - exp(yNN))*datasize/batchsize\
                            +const
    #   ----------------------------------------------------------------------
    ##########################################################################
    grad_lb = toVector(grad_g_lik_store.mean(axis=0))
    
    gradient_lambda = vbNaturalGradient(b,c,grad_lb,isotropic)
    
    return gradient_lambda, beta, grad_g_lik_store, lb_iter

def deepGLMTrain(X_train,y_train,est):
# Traing a deepGLM model with continuous reponse y.
# Bayesian Adaptive Group Lasso is used on the first-layer weights; no
# regularization is put on the rest. sigma2 and tau are updated by
# mean-field VB. Inverse gamma prior is used for sigma2
# INPUT
#   X_train, y_train:           Training data (continuous response)
#   X_validation, y_validation: Validation data
#   n_units:                    Vector specifying the numbers of units in
#                               each layer
#   batchsize:                  Mini-batch size used in each iteration
#   eps0:                       Constant learning rate
#   isotropic:                  True if isotropic structure on Sigma is
#                               used, otherwise rank-1 structure is used
# OUTPUT
#   W_seq:                      The optimal weights upto the last hidden
#                               layer
#   beta                        The optimal weights that connect last hidden layer to the output
#   mean_sigma2                 Estimate of sigma2
#   shrinkage_gamma_seq         Update of shrinkage parameters over
#                               iteration
#   MSE_DL                      Mean squared error over iteration
#
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    # Extract training data and settings from input struct
    X_val = est.data.Xval
    y_val = est.data.yval
    n_units = est.network.astype(int)
    batchsize = est.batchsize
    lrate = est.lrate
    isotropic = est.isIsotropic
    S = est.S                   # Number of Monte Carlo samples to estimate the gradient
    tau = est.tau               # Threshold before reducing constant learning rate eps0
    grad_weight = est.momentum  # Weight in the momentum 
    cScale = est.c              # Random scale factor to initialize b,c
    patience = est.patience     # Stop if test error not improved after patience_parameter iterations
    epoch = est.epoch           # Number of times learning algorithm scan entire training data
    verbose = est.verbose
    distr = est.dist
    lbFlag = est.lowerbound         # Lowerbound flag
    LBwindow = est.windowSize
    seed = est.seed
    
    if (not np.isnan(seed)):
        print('set seed')
        np.random.seed(seed)

    # Data merge for mini-batch sampling
    data = np.concatenate([y_train, X_train], axis=1)       
    datasize = len(y_train)
    num1Epoch = round(datasize/batchsize)    # Number of iterations per epoch

    # Network parameters
    L = len(n_units)        # Number of hidden layers
    p = X_train.shape[1] - 1     # Number of covariates
    W_seq = [[] for t in range(L)]          # Cells to store weight matrices
    index_track = np.zeros(L).astype(int)   # Keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
    index_track[0] = n_units[0]*(p+1)            # Size of W1 is m1 x (p+1) with m1 number of units in the 1st hidden layer 
    W1_tilde_index = np.arange(n_units[0]+1,index_track[0]+1) # Index of W1 without biases, as the first column if W1 are biases
    w_tilde_index = np.array([]) # indices of non-biase weights, excluding W1, for l2-regulization prior
    for j in range(1,L):
        index_track[j] = index_track[j-1] + n_units[j]*(n_units[j-1]+1)
        w_tilde_index = np.concatenate([w_tilde_index,\
                                        np.arange((index_track[j-1]+ n_units[j]+1),index_track[j]+1)])

    d_w = index_track[L-1]      # Total number of weights up to (and including) the last layer
    d_beta = n_units[L-1]+1    # Dimension of the weights beta connecting the last layer to the output
    d_theta = d_w+d_beta     # Total number of parameters
    w_tilde_index = np.concatenate([w_tilde_index,np.arange(d_w+2,d_theta+1)])
    w_tilde_index = (w_tilde_index-1).astype(int)
    d_w_tilde = len(w_tilde_index)

    # Initialise weights and set initial mu equal to initial weights
    layers = np.concatenate([[X_train.shape[1]], n_units, [1]], axis=0).astype(int)  # Full structure of NN -> [input,hidden,output]
    weights = nnInitialize(layers)
    mu = []
    for i in range(len(layers)-1):
        mu = np.concatenate([mu,weights[i].transpose().flatten()], axis=0)
    mu = mu.reshape([len(mu),1])

    # Initialize b and c
    # b = normrnd(0,cScale,d_theta,1);
    b = cScale*np.random.rand(d_theta.astype(int),1)
    if isotropic:
        c = cScale
    else:
        c = cScale*np.ones([d_theta.astype(int),1])

    # Initialize lambda
    lambda_ = np.concatenate([mu, toVector(b), toVector(c)], axis=0)

    W1 = my_reshape(mu[np.arange(index_track[0])], n_units[0], p+1)

    W_seq[0] = W1 
    for j in range(1, L):
#         index = np.arange(index_track[j-1]+1, index_track[j]+1)
        index = np.arange(index_track[j-1], index_track[j])
        Wj = my_reshape(mu[index], n_units[j], n_units[j-1]+1)
        W_seq[j] = Wj 

    beta = mu[np.arange(d_w, d_theta)]

    # Get mini-batch
    idx = np.random.permutation(datasize)[:batchsize]
    minibatch = data[idx,:]
#     minibatch = np.copy(data) # Test
    
    y = toVector(minibatch[:,0])
    X = minibatch[:,1:]
    
    # Hyperparameters for inverse-Gamma prior on sigma2 if y~Nomal(0,sigma2)
    mean_sigma2_save = []
    if (distr == 'normal'):
        alpha0_sigma2 = 10; 
        beta0_sigma2 = (alpha0_sigma2-1)*np.std(y, ddof=1) 
        alpha_sigma2 = alpha0_sigma2 + len(y_train)/2  # Optimal VB parameter for updating sigma2 
        beta_sigma2 = alpha_sigma2                     # Mean_sigma2 and mean_sigma2_inverse are 
                                                       # Initialised at small values 1/2 and 1 respectively  
        mean_sigma2_inverse = alpha_sigma2/beta_sigma2
        mean_sigma2 = beta_sigma2/(alpha_sigma2-1)
        mean_sigma2_save.append(mean_sigma2)

    # Compute prediction loss if not using lowerbound for validation
    MSE_DL = []
    PPS_DL = []
    if (not lbFlag):
        if (distr == 'normal'):
            PPS_current, MSE_current = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr,mean_sigma2);
    #         disp(['Initial MSE: ',num2str(MSE_current)]);
        else:
            PPS_current, MSE_current = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr);
    #         disp(['Initial PPS: ',num2str(PPS_current)]);

        MSE_DL.append(MSE_current)
        PPS_DL.append(PPS_current)

    # Calculations for group Lasso coefficients
    shrinkage_gamma = .01*np.ones([p,1]) # Initialise gamma_beta, the shrinkage parameters
    shrinkage_l2 = .01              # Hype-parameter for L2 prior
    mu_tau = np.zeros([p,1])             # Parameters for the auxiliary tau_j
    mu_matrixW1_tilde = my_reshape(mu[W1_tilde_index-1], n_units[0], p)
    b_matrixW1_tilde = my_reshape(b[W1_tilde_index-1], n_units[0], p)
    if isotropic:
        for j in range(p):
            mean_column_j_tilde = np.dot(mu_matrixW1_tilde[:,j], mu_matrixW1_tilde[:,j])+\
                np.dot(b_matrixW1_tilde[:,j],b_matrixW1_tilde[:,j])+c**2*n_units[0]
            mu_tau[j] = shrinkage_gamma[j]/np.sqrt(mean_column_j_tilde)        

        lambda_tau = np.power(shrinkage_gamma,2)
    else:
        c_matrixW1_tilde = my_reshape(c[W1_tilde_index-1], n_units[0], p)                             
        for j in range(p):
            mean_column_j_tilde = np.dot(mu_matrixW1_tilde[:,j], mu_matrixW1_tilde[:,j])+\
                np.dot(b_matrixW1_tilde[:,j], b_matrixW1_tilde[:,j])+np.sum(np.power(c_matrixW1_tilde[:,j], 2))
            mu_tau[j] = shrinkage_gamma[j]/np.sqrt(mean_column_j_tilde)

        lambda_tau = np.power(shrinkage_gamma, 2)

    mean_inverse_tau = mu_tau              # VB mean <1/tau_j>
    shrinkage_gamma_seq = shrinkage_gamma  
    mean_tau = 1/mu_tau + 1/lambda_tau
    m = n_units[1]

    # Prepare to calculate lowerbound
    if (lbFlag):
        if (distr == 'normal'):
            const = alpha0_sigma2*np.log(beta0_sigma2) - gammaln(alpha0_sigma2)\
                    -0.5*p*n_units[0]*np.log(2*np.pi)-0.5*d_w_tilde*np.log(2*np.pi)\
                    -p*gammaln((n_units[0]+1)/2)-0.5*datasize*np.log(2*np.pi)\
                    +p/2*np.log(2*np.pi)+0.5*d_theta*np.log(2*np.pi)+d_theta/2
        else:
            const = -0.5*p*n_units[0]*np.log(2*np.pi)-0.5*d_w_tilde*np.log(2*np.pi)\
                    -p*gammaln((n_units[0]+1)/2)+p/2*np.log(2*np.pi)\
                    +0.5*d_theta*np.log(2*np.pi)+d_theta/2

        W1 = my_reshape(mu[np.arange(index_track[0])], n_units[0], p+1)
        W_seq[1] = W1 
        for j in range(1,L):
            index = np.arange(index_track[j-1],index_track[j])
            Wj = my_reshape(mu[index], n_units[j], n_units[j-1]+1)
            W_seq[j]= Wj

        beta = mu[np.arange(d_w,d_theta)]
        mu_w_tilde = mu[w_tilde_index]
        b_w_tilde = b[w_tilde_index]
        c_w_tilde = c[w_tilde_index]
    #     mean_w_tilde = np.dot(mu_w_tilde, np.dot(mu_w_tilde+b_w_tilde,b_w_tilde))\
    #                     + np.sum(np.power(c_w_tilde,2))
        mean_w_tilde = np.dot(mu_w_tilde.transpose(), mu_w_tilde) \
                        + np.dot(b_w_tilde.transpose(),b_w_tilde) + sum(np.power(c_w_tilde,2))
        iter = 0
        logdet, constMean = vbLowerBound(beta_sigma2, alpha_sigma2,
                                        b, c, distr, p,
                                        alpha0_sigma2, beta0_sigma2,
                                        mean_sigma2_inverse, n_units, 
                                        shrinkage_gamma, mean_tau,
                                        datasize, lambda_tau,
                                        d_w_tilde, shrinkage_l2,
                                        mean_w_tilde, mean_column_j_tilde,
                                        mean_inverse_tau)
    #     print('Initial LB: %s' % str(lb[iter]))
    grad_g_lik_store = zeros([S,3*d_theta])
    gradient_lambda, beta, grad_g_lik_store, lb_iter = vbGradientLogLB(S, d_theta, mu, L, index_track, n_units, p,
               beta, X, y, datasize, distr, mean_sigma2_inverse,
               b, c, isotropic, mean_inverse_tau, shrinkage_l2,
               d_w, lbFlag, constMean, batchsize, const,
               grad_g_lik_store)    
    
    gradient_bar = gradient_lambda
    
    # lb = zeros([num1Epoch*epoch])
    if (lbFlag):
        lb = [lb_iter.mean()/datasize]
    #     lb[iter] = lb_iter.mean()/datasize
        print('Initial LB: %s' % lb[iter])
    #--------------------------------------------------------------------------
        
    ## Training Phase
    # Prepare parameters for training
    idxEpoch = 0          # Index of current epoch
    iter = 0              # Index of current iteration
    stop = False          # Stop flag for early stopping
    lambda_best = lambda_  # Store optimal lambda for output
    idxPatience = 0       # Index of number of consequent non-decreasing iterations
                           # for early stopping
    # lb_bar = zeros(num1Epoch*epoch)
    lb_bar = []
    print('---------- Training Phase ----------')

    while stop!=True:
        iter = iter + 1

        ## ------------------Natural Gradient Calculation----------------------
        # Get mini-batch
        idx = np.random.permutation(datasize)[:batchsize]
        minibatch = data[idx,:]
#         minibatch = np.copy(data)

        y = toVector(minibatch[:,0])
        X = minibatch[:,1:]

        # Calculate expected terms of lowerbound
        if (lbFlag):
            logdet, constMean = vbLowerBound(beta_sigma2, alpha_sigma2,
                                            b, c, distr, p,
                                            alpha0_sigma2, beta0_sigma2,
                                            mean_sigma2_inverse, n_units, 
                                            shrinkage_gamma, mean_tau,
                                            datasize, lambda_tau,
                                            d_w_tilde, shrinkage_l2,
                                            mean_w_tilde, mean_column_j_tilde,
                                            mean_inverse_tau)           

        # Calculate Natural Gradient
        print('Calculate Natural Gradient');
        gradient_lambda, beta, grad_g_lik_store, lb_iter = vbGradientLogLB(S, d_theta, mu, L, index_track, n_units, p,
               beta, X, y, datasize, distr, mean_sigma2_inverse,
               b, c, isotropic, mean_inverse_tau, shrinkage_l2,
               d_w, lbFlag, constMean, batchsize, const,
               grad_g_lik_store)

        # Get lowerbound in the current iteration
        if (lbFlag):
            lb += [np.mean(lb_iter)/datasize]
        #----------------------------------------------------------------------

        ## ------------------Stochastic gradient ascend update-----------------
        # Prevent exploding Gradient
        grad_norm = norm(gradient_lambda)
        norm_gradient_threshold = 100
        if norm(gradient_lambda) > norm_gradient_threshold:
            gradient_lambda = (norm_gradient_threshold/grad_norm)*gradient_lambda
        
        # Momentum gradient
        gradient_bar_old = gradient_bar
        gradient_bar = grad_weight*gradient_bar + (1 - grad_weight)*gradient_lambda     

        # Adaptive learning rate
        if iter > tau:
            stepsize = lrate*tau/iter
        else:
            stepsize = lrate

        # Gradient ascend
        lambda_ = lambda_ + stepsize*gradient_bar
        # Restore model parameters from variational parameter lambda
        mu = toVector(lambda_[:d_theta,0])
        b = toVector(lambda_[d_theta:2*d_theta,0])
        c = toVector(lambda_[2*d_theta:])
        W1 = my_reshape(mu[:index_track[0]], n_units[0],p+1)
        W_seq[0] = W1
        for j in range(1,L):
            index = arange(index_track[j-1], index_track[j])
            Wj = my_reshape(mu[index], n_units[j],n_units[j-1]+1)
            W_seq[j] = Wj

        beta = toVector(mu[d_w:d_theta])
        
        #----------------------------------------------------------------------

        ## ---------------- Update tau and shrinkage parameters----------------    
        if iter % 1 == 0:
            mu_matrixW1_tilde = my_reshape(mu[W1_tilde_index-1],n_units[0],p)
            b_matrixW1_tilde = my_reshape(b[W1_tilde_index-1],n_units[0],p)
            mean_column_j_tilde = zeros([p,1])
            if isotropic:
                for j in range(p):
                    mean_column_j_tilde[j] = dot(mu_matrixW1_tilde[:,j].transpose(),mu_matrixW1_tilde[:,j])\
                                                + dot(b_matrixW1_tilde[:,j].transpose(), b_matrixW1_tilde[:,j])\
                                                + c^2*n_units[0]
                    mu_tau[j] = shrinkage_gamma[j]/sqrt(mean_column_j_tilde[j])
                    lambda_tau[j] = shrinkage_gamma[j]^2
            else:
                c_matrixW1_tilde = my_reshape(c[W1_tilde_index-1],n_units[0],p)
                for j in range(p):
                    mean_column_j_tilde[j] = dot(mu_matrixW1_tilde[:,j],mu_matrixW1_tilde[:,j])\
                                                 + dot(b_matrixW1_tilde[:,j],b_matrixW1_tilde[:,j])\
                                                 + np.sum(power(c_matrixW1_tilde[:,j],2))
                    mu_tau[j] = shrinkage_gamma[j]/sqrt(mean_column_j_tilde[j])
                    lambda_tau[j] = shrinkage_gamma[j]**2

            mean_inverse_tau = np.copy(mu_tau)
            mean_tau = 1/mu_tau + 1/lambda_tau
            shrinkage_gamma = sqrt((n_units[0] + 1)/mean_tau)
            shrinkage_gamma_seq = concatenate([shrinkage_gamma_seq,shrinkage_gamma], axis=1)

            mu_w_tilde = mu[w_tilde_index]
            b_w_tilde = b[w_tilde_index] 
            c_w_tilde = c[w_tilde_index]
            mean_w_tilde = dot(mu_w_tilde.transpose(),mu_w_tilde) + dot(b_w_tilde.transpose(),b_w_tilde)\
                                                 + np.sum(power(c_w_tilde,2))
    #         shrinkage_l2 = length(w_tilde_index)/mean_w_tilde

        ## ------Update VB posterior for sigma2, which is inverse Gamma -------
        # if y ~ N(0,sigma2)    
        if distr == 'normal':
            if iter % 1 == 0:     
                sum_squared = sumResidualSquared(y_train,X_train,W_seq,beta)
                beta_sigma2 = beta0_sigma2 + sum_squared/2
                mean_sigma2_inverse = alpha_sigma2/beta_sigma2
                mean_sigma2 = beta_sigma2/(alpha_sigma2 - 1)
                mean_sigma2_save.append(mean_sigma2)
        #----------------------------------------------------------------------

        ## ----------------------------Validation------------------------------
        # If using lowerbound for validation
        if (lbFlag):
            # Storing lowerbound moving average values
            if (iter >= LBwindow):
                lb_bar += [mean(lb[iter-LBwindow:iter])];
#                 lb_bar[iter-LBwindow] = mean(lb[iter-LBwindow:iter]);
                if lb_bar[-1] >= max(lb_bar):
                    lambda_best = lambda_
                    idxPatience = 0
                else:
                    idxPatience = idxPatience + 1
#                     disp(['idxPatience: ',num2str(idxPatience)])

        # If using MSE/Accuracy for validation
        else: 
            if distr == 'normal':
                PPS_current,MSE_current = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr,mean_sigma2)
            else:
                PPS_current,MSE_current = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr)

            MSE_DL[iter] = MSE_current
            PPS_DL[iter] = PPS_current

            if PPS_DL[iter] >= PPS_DL[iter]:
                gradient_bar = gradient_bar_old

            if PPS_DL[iter] <= min(PPS_DL):
                lambda_best = lambda_
                idxPatience = 0
            else:
                idxPatience = idxPatience + 1
#                 disp(['idxPatience: ',num2str(idxPatience)])

        # Early stopping
        if (idxPatience > patience) or (idxEpoch > epoch): 
            stop = True
        #----------------------------------------------------------------------

        ## ------------------------------Display-------------------------------
        # Display epoch index whenever an epoch is finished
        if iter % num1Epoch == 0:
            idxEpoch = idxEpoch + 1

        # Display training results after each 'verbose' iteration
        if ((verbose) and (iter % verbose == 0)):
            if (lbFlag):     # Display lowerbound
    #             disp(['Epoch: ',num2str(idxEpoch)]);

                if (iter>=LBwindow):
                    print('Iter: %s - Epoch: %s - Current LB: %s' % (iter, idxEpoch, lb_bar[iter-LBwindow]))
                else:
                    print('Iter: %s - Epoch: %s - Current LB: %s' % (iter, idxEpoch, lb[iter]))
            else:       # Or display MSE/Accuracy
                if distr=='binomial':
                    print('Current PPS: %s' % PPS_current)
                else:
                    print('Current MSE: %s' % MSE_current)
#             print('b: %s' % (1e3*b).transpose())
        print('grad: %s' % (gradient_bar*1e3).transpose())

        #----------------------------------------------------------------------
            
    ## --------------------------Display Training Results----------------------
    print('---------- Training Completed! ----------')
    print('Number of iteration: %s' % iter)
    if (lbFlag):
        print('LBBar best: %s' % max(lb_bar))
    else:
        print('PPS best: %s' % min(PPS_DL))
        print('MSE best: %s' % min(MSE_DL))

    ## ----------------------Store training output-----------------------------
    lambda_ = lambda_best;
    mu = toVector(lambda_[:d_theta,0])
    b = toVector(lambda_[d_theta:2*d_theta,0])
    c = toVector(lambda_[2*d_theta:])
    if isotropic:              # For isotropic structure
        SIGMA = dot(b,b.transpose()) + c**2*eyes(d_theta)
    else:
        SIGMA = dot(b,b.transpose()) + diag(power(c,2))

    W1 = my_reshape(mu[:index_track[0]], n_units[0], p+1)
    W_seq[0] = W1
    for j in range(1,L):
        index = range(index_track[j-1], index_track[j])
        Wj = my_reshape(mu[index], n_units[j], n_units[j-1]+1)
        W_seq[j] = Wj

    beta = mu[d_w:d_w+d_beta]

    # Store output in a struct
    est.out.weights = W_seq; 
    est.out.beta = beta;
    est.out.shrinkage = shrinkage_gamma_seq;
    est.out.iteration = iter;
    est.out.vbMU = mu.flatten();            # Mean of variational distribution of weights
    est.out.b = b;
    est.out.c = c;
    est.out.vbSIGMA = SIGMA;      # Covariance matrix of variational distribution 
                                  # of weights
    est.out.nparams = d_theta;    # Number of parameters     
    est.out.indexTrack = index_track;
    est.out.muTau = mu_tau;

    if distr=='normal':
        est.out.sigma2Alpha = alpha_sigma2;
        est.out.sigma2Beta = beta_sigma2;
        est.out.sigma2Mean = mean_sigma2_save[-1]
        est.out.sigma2MeanIter = mean_sigma2_save;

    if (lbFlag):
        est.out.lbBar = lb_bar[1:];
        est.out.lb = lb;
    else:
        if (distr == 'binomial'):
            est.out.accuracy = MSE_DL;
        else:
            est.out.mse = MSE_DL;

        est.out.pps = PPS_DL;
            
            
    return est

def plotShrinkage(ShrinkageCoef,opt):
#PLOTSHRINKAGE Plot shrinkage coefficient of Group Lasso regularization
#
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

# Do not plot intercept coefficient
# ShrinkageCoef = ShrinkageCoef(2:end,:);

    TextTitle = opt.title
    labelX = opt.labelX
    labelY = opt.labelY
    linewidth = opt.linewidth
    color = opt.color

    numCoeff = ShrinkageCoef.shape[0]   # Number of shrinkage coefficients
    font = {'size'   : 13}
    matplotlib.rc('font', **font)

    # Define default settings
    if TextTitle == '':
        TextTitle = 'Shrinkage Coefficients';
    
    if labelX == '':
        labelX = 'Iteration'

    # Plot
    plt.figure(figsize=(15,8))
    plt.plot(ShrinkageCoef.transpose());
    plt.grid()
    plt.title(TextTitle, fontdict = {'fontsize': 20})
    plt.xlabel(labelX, fontdict = {'fontsize': 15})
    plt.ylabel(labelY, fontdict = {'fontsize': 15})
    Ytext = ShrinkageCoef[:,-1]  # Y coordination of text, different for coefficients
    Xtext = ShrinkageCoef.shape[1] # X coordination of text, same for all coefficients 
    for i in range(numCoeff):
        plt.text(Xtext, Ytext[i],
                 s = r"$\gamma_{" + str(i) + r"}$",
                 fontdict = {'fontsize': 13})
    plt.show()
    
def plotInterval(predMean,predInterval,opt,
                 color = 'red', style='shade', ytrue=[]):
#PLOTINTERVAL Plot prediction interval for test data
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    textTitle = opt.title;
    labelX = opt.labelX;
    labelY = opt.labelY;
    linewidth = opt.linewidth;

    # Define some default texts
    if (textTitle==''):
        textTitle = 'Prediction Interval on Test Data'

    if (labelX==''):
        labelX = 'Observation'

    lower = predInterval[:,0]
    upper = predInterval[:,1]
    t = list(range(len(predMean)))

    plt.figure(figsize=(15,8))
    
    if style == 'shade':       # Plot prediction interval in shade style
        plt.scatter(t, predMean, label='Prediction values')
        plt.fill_between(t, upper, lower, alpha=0.1)
        if not (ytrue==[]):
            plt.scatter(t, ytrue, c=color, label='True values')
        
        plt.legend()
        plt.grid()
        plt.title(textTitle, fontdict = {'fontsize': 18})
#         case 'boundary'   % Plot prediction interval in boundary style 
#             plot(t,predMean,'LineWidth',linewidth,'Color',color);
#             hold on
#             plot(t,upper,'--r',t,lower,'--r');
#             grid on
#             title('Prediction Interval on Test Data', 'FontSize',18)
#             xlabel('Observation')
#             hold off
#         case 'bar'        % Plot prediction interval in bar style 
#             err = (upper-lower)/2;
#             errorbar(predMean,err);
#             grid on
#             hold on
#             plot(predMean,'Color','red','LineWidth',2);
#             title('Prediction Interval on Test Data', 'FontSize',18)
#             xlabel('Observation')
#             hold off

def deepGLMplot(type_, Pred, 
                TextTitle='', labelX='', labelY='', 
                linewidth=2, color='red', style='shade', 
                npoint=50, order='ascend', y=[],legendText={}):
#DEEPGLMPLOT Plot analytical figures for deepGLM estimation and prediction
#
#   DEEPGLMPLOT(TYPE, DATA) Plots data specified in data1 according the type 
#   specified by TYPE. TYPE can be one of the following options:
#
#      'Shrinkage'        Plot stored values of group Lasso coefficient during 
#                         training phase. If this type is specify, DATA is the 
#                         output structure mdl from DEEPGLMFIT or user can manually 
#                         extract mdl.out.shrinkage field from mdl an use as input 
#                         argument data.
#      'Interval'         Plot prediction interval estimation for test data. 
#                         If this type is specified, DATA is the output structure 
#                         Pred from DEEPGLMPREDICT.  
#      'ROC'              Plot ROC curve for prediction from binary test data. 
#                         If this type is specify, data is a matrix where the 
#                         first column is a vector of target responses and the 
#                         second column is the predicted vector yProb extract 
#                         from output structure Pred of deepGLMpredict. If you 
#                         want to plot different ROC curves, add more probability 
#                         columns to data.  
#
#   DEEPGLMPLOT(TYPE, DATA, NAME, VALUE) Plots data specified in DATA according 
#   the type specified by type with one of the following NAME/VALUE pairs:
#
#      'Nsample'          This option only available when type is Interval. 
#                         'Nsample' specifies number of test observations 
#                         randomly selected from test data to plot prediction 
#                         intervals. 
#      'Title'            Title of the figure   
#      'Xlabel'           Label of X axis  
#      'Ylabel'           Label of Y axis 
#      'LineWidth'        Line width of the plots
#      'Legend'           Legend creates a legend with descriptive labels for 
#                         each plotted data series
#
#   Example: See deepGLMNormalExample.mlx and deepGLMBinomialExample.mlx
#   in the Examples folder
#
#   See also DEEPGLMFIT, DEEPGLMPREDICT
#
#   Copyright 2018:
#       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
#       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
#      
#   https://github.com/VBayesLab/deepGLM
#
#   Version: 1.0
#   LAST UPDATE: May, 2018

# Store plot options to a structure
    class optClass():
        def __init__(self, title, lableX, labelY, linewidth, color):
            self.title = TextTitle
            self.labelX = labelX
            self.labelY = labelY
            self.linewidth = linewidth
            self.color = color
    opt = optClass(TextTitle, labelX, labelY, linewidth, color)
                                            
    if type_ == 'Shrinkage':
            plotShrinkage(Pred,opt);
    elif type_ == 'Interval':
            yhat = Pred.yhatMatrix
            yhatInterval = Pred.interval
            predMean = yhat.mean(axis=0)
            # If test data have more than 100 rows, extract randomly 100 points to draw
            if (len(predMean) >= npoint):
                idx = permutation(len(yhatInterval))[:npoint]
                intervalPlot = yhatInterval[idx,:]
                yhatMeanPlot = predMean[idx].transpose()
                if not (y==[]):
                    ytruePlot = y[idx]#.transpose()
            else:
                yhatMeanPlot = predMean.transpose()
                intervalPlot = yhatInterval
                ytruePlot = y
            
            # Sort data
            sortIdx = argsort(yhatMeanPlot)
            if order != 'ascend':
                sortIdx = sortIdx[::-1]
            yhatMeanPlot = yhatMeanPlot[sortIdx]
            intervalPlot = intervalPlot[sortIdx,:]
            if (y==[]):
                ytruePlot = [];
            else:
                ytruePlot = ytruePlot[sortIdx]
                
            plotInterval(yhatMeanPlot,intervalPlot,opt,
                        ytrue=ytruePlot,
                        style=style);
#     elif type_ == 'ROC':
#             if(~isnumeric(y))
#                 disp('Target should be a column of binary responses!')
#                 return
#             else
#                 % Plot single ROC
#                 if(size(Pred,2)==1)
#                     [tpr,fpr,~] = roc(y',Pred');
#                     plot(fpr,tpr,'LineWidth',linewidth);
#                     grid on
#                     title(TextTitle,'FontSize',20);
#                     xlabel(labelX,'FontSize',15);
#                     ylabel(labelY,'FontSize',15);
#                 % Plot multiple ROC
#                 else
#                     tpr = cell(1,size(Pred,2));
#                     fpr = cell(1,size(Pred,2));
#                     for i=1:size(Pred,2)
#                         [tpr{i},fpr{i},~] = roc(y',Pred(:,i)');
#                         plot(fpr{i},tpr{i},'LineWidth',linewidth);
#                         grid on
#                         hold on
#                     end
#                     title(TextTitle,'FontSize',20);
#                     xlabel(labelX,'FontSize',15);
#                     ylabel(labelY,'FontSize',15);
#                     legend(legendText{1},legendText{2});
                            
def predictionInterval(mdl,X,zalpha):
#CONFIDENTINTERVAL Interval estimation for test data using deepGLM
#
#   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
#   Nguyen (nghia.nguyen@sydney.edu.au)
#   
#   http://www.xxx.com
#
#   Version: 1.0
#   LAST UPDATE: April, 2018

    # Load deepGLM params from struct
    Nsample = mdl.Nsample
    MU = mdl.out.vbMU
    SIGMA = mdl.out.vbSIGMA
    n_units = mdl.network.astype(int)
    index_track = mdl.out.indexTrack
    alpha_sigma2 = mdl.out.sigma2Alpha
    beta_sigma2 = mdl.out.sigma2Beta

    # Calculate network parameters 
    L = len(n_units)        # Number of hidden layers
    p = X.shape[1] - 1      # Number of covariates
    d_beta = n_units[L-1] + 1 
    d_w = index_track[L-1]

    yhat = zeros([Nsample, X.shape[0]]) # Predicted values of test data
    nnOut = zeros([Nsample, X.shape[0]])      # Output of NN
    W_seq = [[] for i in range(L)]
    for i in range(Nsample):
        # Generate samples of theta from Normal distribution
        theta_i = multivariate_normal(MU,SIGMA)
        # Generate samples of sigma from IG distribution
        sigma2_i = 1/gamma(alpha_sigma2, 1/beta_sigma2)

        # For each generated theta, restore neuron net structure 
        W1 = my_reshape(theta_i[:index_track[0]], n_units[0], p+1)
        W_seq[0] = W1 
        for j in range(1,L):
            index = arange(index_track[j-1], index_track[j])
            Wj = my_reshape(theta_i[index], n_units[j], n_units[j-1]+1)
            W_seq[j] = Wj
        
        beta = theta_i[arange(d_w, d_w+d_beta)].transpose()

        # Calculate neuron network output
        nnOut[i,:] = nnFeedForward(X,W_seq,beta)

        # Calculate p(y|theta_i,sigma_i,X)
        yhat[i,:] = normal(nnOut[i,:], sqrt(sigma2_i))

    # 95% confidence interval
    yhatLCL = mean(yhat, axis=0) - zalpha*std(yhat, axis=0)
    yhatUCL = mean(yhat, axis=0) + zalpha*std(yhat, axis=0)
    yhatInterval = concatenate([toVector(yhatLCL), toVector(yhatUCL)], axis=1)
    
    class predIntervalClass():
        yhatMC = np.nan
        interval = np.nan
        
    predInterval = predIntervalClass()
    predInterval.yhatMC = yhat
    predInterval.interval = yhatInterval

    return predInterval

def deepGLMpredict(mdl,X_,
                   y=[], alpha=0, Nsample=1000, intercept=True):
#DEEPGLMPREDICT Make prediction from a trained deepGLM model
#
#   OUT = DEEPGLMPREDICT(MDL,XTEST) predict responses for new data XTEST using 
#   trained deepGLM structure MDL (output from DEEPGLMFIT) 
#
#   OUT = DEEPGLMPREDICT(MDL,XTEST,NAME,VALUE) predicts responses with additional 
#   options specified by one or more of the following name/value pairs:
#
#      'ytest'            Specify column of test responses. If this option 
#                         is specified with true response column of new 
#                         observations, deepGLMpredict will return prediction 
#                         scores (PPS, MSE or Classification Rate) using true 
#                         responses column vector ytest
#      'Interval'         Return prediction interval estimation for observations 
#                         in test data Xtest. By default, this predictive 
#                         interval capability is disable ('Interval' is 0). 
#                         Must be an positive number.   
#      'Nsample'          Number of samples generated from posterior distribution 
#                         of model parameters used to make prediction interval 
#                         estimation for test data. Must be a positive integer
#   Example:
#      Fit a deepGLM model for Direcmarketing data set. All of the
#      exampled data are located inside /Data folder of installed package. 
#      In order to use the sample dataset, user must add this Data folder
#      to Matlab path or explicitly direct to Data folder in 'load'
#      function
#
#      load('DirectMarketing.mat')
#      mdl = deepGLMfit(X,y,...                   % Training data
#                      'Network',[5,5],...        % Use 2 hidden layers
#                      'Lrate',0.01,...           % Specify learning rate
#                      'Verbose',10,...           % Display training result each 10 iteration
#                      'BatchSize',size(X,1),...  % Use entire training data as mini-batch
#                      'MaxEpoch',10000,...       % Maximum number of epoch
#                      'Patience',50,...          % Higher patience values could lead to overfitting
#                      'Seed',100);               % Set random seed to 100
#    
#      Pred = deepGLMpredict(mdl,X_test,...
#                           'ytest',y_test);
#      disp(['PPS on test data: ',num2str(Pred.pps)])
#      disp(['MSE on test data: ',num2str(Pred.mse)])
#   
#   For more examples, check EXAMPLES folder
#
#   See also DEEPGLMFIT, DEEPGLMPLOT
#
#   Copyright 2018:
#       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
#       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
#      
#   https://github.com/VBayesLab/deepGLM
#
#   Version: 1.0
#   LAST UPDATE: May, 2018


    # Load deepGLM params from struct
    W_seq = mdl.out.weights;
    beta = mdl.out.beta;
    distr = mdl.dist;


    # If y test is specified, check input
    if len(y)!=0:
        if y.shape[0] != X_.shape[0]:
            raise(deepGLMmsg('deepglm:InputSizeMismatchX'))
        
        if y.shape[1] != 1:
            raise(deepGLMmsg('deepglm:InputSizeMismatchY'))

    # Add column of 1 to X if intercept is true
    if (intercept):
        X = concatenate([ones([X_.shape[0],1]),X_], axis=1)

    # Store Nsample to deepGLMfit
    mdl.Nsample = Nsample

    # Calculate neuron network output
    nnet_output = nnFeedForward(X,W_seq,beta)

    class outClass():
        yhat = np.nan
        mse = np.nan
        pps = np.nan
        interval = np.nan
        yhatMatrix = np.nan
        yNN = np.nan
        yProb = np.nan
        yhat = np.nan
    
    out = outClass()
    
    if distr == 'normal':
        out.yhat = nnet_output;    # Prediction for continuous response
        # If ytest if provided, then calculate pps and mse
        if len(y)!=0:
            sigma2 = mdl.out.sigma2Mean
            mse = mean(power(y-nnet_output, 2))
            pps = 1/2*log(sigma2) + 1/2/sigma2*mse
            out.mse = mse
            out.pps = pps
            
        # Calculate confidence interval if required
        if (alpha!=0):
            interval = predictionInterval(mdl,X,alpha)
            out.interval = interval.interval
            out.yhatMatrix = interval.yhatMC
        

    elif distr == 'binomial':
        out.yNN = nnet_output
        out.yProb = exp(nnet_output)/(1 + exp(nnet_output))
        y_pred = (nnet_output>0).astype('double')   # Prediction for binary response
        out.yhat = y_pred
        # If ytest if provided, then calculate pps and mse
        if len(y)!=0:
            pps = mean(-y*nnet_output + log(1 + exp(nnet_output)))
            cr = mean(y==y_pred)    # Miss-classification rate
            out.pps = pps
            out.accuracy = cr
    elif distr == 'poisson':
        out.yNN = nnet_output
        y_pred = exp(nnet_output)    # Prediction for poisson response
        out.yhat = y_pred
        if len(y)!=0:
            pps = mean(-y*nnet_output + exp(nnet_output))
            mse = mean(power(y-y_pred, 2))
            out.mse = mse
            out.pps = pps
            
    return out

