function est = deepGLMfit(X,y,varargin)
%DEEPGLM Traing a deepGLM model. DeepGLM is a flexible version of Generalized 
% Liner Model where Deep Feedforward Network is used to automatically choose 
% transformations for the raw covariates. Bayesian Adaptive Group Lasso is 
% used on the first-layer weights; a Ridge-type prior is imposed to the rests. 
% sigma2 and tau are updated by mean-field VB. Inverse gamma prior is used for 
% sigma2
% 
%   MDL = DEEPGLMFIT(X,Y) fits a deepGLM model using the design matrix X and 
%   response vector Y, and returns an output structure mdl to make prediction 
%   on a test data. By default, if 'distribution' option is not specified, 
%   deepGLMfit will treat response variable y as normal distributed variable.
%
%   MDL = DEEPGLMFIT(X,Y,NAME,VALUE) fit a deepGLM model with additional options
%   specified by one or more of the following NAME/VALUE pairs:
%
%      'Distribution'     Name of the distribution of the response, chosen
%                         from the following:
%                 'normal'             Normal distribution (default)
%                 'binomial'           Binomial distribution
%                 'poisson'            Poisson distribution
%      'Network'          Deep FeedforwardNeuron Network structure for deepGLM. 
%                         In the current version, deepGLM supports only 1 node 
%                         for the output layer, users just need to provide a 
%                         structure for hidden layers in an array where each 
%                         element in the array is the 
%                         number of nodes in the corresponding hidden layer.
%      'Lrate'            Vector of integer or logical indices specifying
%                         the variables in TBL or the columns in X that
%                         should be treated as categorical. Default is to
%                         treat TBL variables as categorical if they are
%                         categorical, logical, or char arrays, or cell
%                         arrays of strings.
%      'Momentum'         Momentum weight for stochastic gradient ascend. 
%                         The momentum determines the contribution of the 
%                         gradient step from the previous iteration to the 
%                         current iteration of training. It must be a value 
%                         between 0 and 1, where 0 will give no contribution 
%                         from the previous step, and 1 will give a maximal 
%                         contribution from the previous step. Must be between 
%                         0 and 1. 
%      'BatchSize'        The size of the mini-batch used for each training 
%                         iteration. Must be a positive integer smaller than 
%                         number of observations of training data
%      'MaxEpoch'         The maximum number of epochs that will be used for 
%                         training. An epoch is defined as the number of 
%                         iterations needed for optimization algorithm to 
%                         scan entire training data. Must be a positive integer.
%      'Patience'         Number of consecutive times that the validation loss 
%                         is allowed to be larger than or equal to the previously 
%                         smallest loss before network training is stopped, 
%                         used as an early stopping criterion. Must be a positive 
%                         integer.
%      'LrateFactor'      Down-scaling factor that is applied to the learning 
%                         rate every time a certain number of iterations has 
%                         passed. Must be a positive integer
%      'S'                The number of samples needed for Monte Carlo 
%                         approximation of gradient of lower bound. Must 
%                         be an positive integer
%      'WindowSize'       Size of moving average window that used to smooth 
%                         the VB lowerbound. Must be an positive integer
%      'Intercept'        Set true (default) to add a column of 1 to predictor 
%                         observation X matrix (play the role as intercept). 
%                         If the data have already included the first '1' column, 
%                         set 'Intercept' to false.
%      'Verbose'          Number of iterations that information on training 
%                         progress will be printed to the command window each 
%                         time. Set to 0 to disable this options. 
%      'Monitor’'         Display monitor window showing the training process 
%                         on a user interface. This is a useful tool to visualize 
%                         training metrics at every iteration. However, using 
%                         this option will slow down training progress because 
%                         of graphical related tasks.
%      'Isotropic'        Set to true if you want to use Isotropic structure 
%                         on Sigma (Variational Covariance matrix). By default, 
%                         deepGLM uses rank-1 structure to factorize Sigma
%      'Seed'             Seeds the random number generator using the nonnegative 
%                         integer. Must be a nonnegative integer.
%
%   Example:
%      Fit a deepGLM model for Direcmarketing data set. All of the
%      exampled data are located inside /Data folder of installed package. 
%      In order to use the sample dataset, user must add this Data folder
%      to Matlab path or explicitly direct to Data folder in 'load'
%      function
%
%      load('DirectMarketing.mat')
%      mdl = deepGLMfit(X,y,...                   % Training data
%                      'Network',[5,5],...        % Use 2 hidden layers
%                      'Lrate',0.01,...           % Specify learning rate
%                      'Verbose',10,...           % Display training result each 10 iteration
%                      'BatchSize',size(X,1),...  % Use entire training data as mini-batch
%                      'MaxEpoch',10000,...       % Maximum number of epoch
%                      'Patience',50,...          % Higher patience values could lead to overfitting
%                      'Seed',100);               % Set random seed to 100
%
%   For more examples, check EXAMPLES folder
%
%   See also DEEPGLMPREDICT, DEEPGLMPLOT
%
%   Copyright 2018:
%       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
%      
%   https://github.com/VBayesLab/deepGLM
%
%   Version: 1.0
%   LAST UPDATE: May, 2018


% Initialize output structure with default setting
est = deepGLMout();

% Check errors input arguments
if nargin < 2
    error(deepGLMmsg('deepglm:TooFewInputs'));
end
if size(y,1) ~= size(X,1)
    error(deepGLMmsg('deepglm:InputSizeMismatchX'));
end
if size(y,2) ~= 1
    error(deepGLMmsg('deepglm:InputSizeMismatchY'));
end
if any(isnan(y)) || any(any(isnan(X))) % Check if data include NaN
    error('NaN values not allowed in input data');
end

if ~isempty(varargin)
    if mod(length(varargin),2)==1 % odd, model followed by pairs
        error(deepGLMmsg('deepglm:ArgumentMustBePair'));
    end
end

%% Parse additional options
paramNames = {'Isotropic'       'S'             'BatchSize'     'Lrate'   ...
              'Initialize'      'Ncore'         'MaxEpoch'      'LRateFactor'     ...
              'Patience'        'Network'       'Distribution'  'Seed'    ...
              'C'               'Bvar'          'Xval'          'Yval'    ...
              'Nval'            'Intercept'     'Momentum'      'Verbose' ...
              'BinaryCutOff’'   'QuasiMC'       'Monitor'       'MuTau'   ...
              'LowerBound'      'WindowSize'};
paramDflts = {est.isIsotropic   est.S           est.batchsize   est.lrate      ...
              est.initialize    est.ncore       est.epoch       est.tau        ...
              est.patience      est.network     est.dist        est.seed       ...
              est.c             est.bvar        est.data.Xval   est.data.yval  ...
              est.nval          est.data.icept  est.momentum    est.verbose    ...
              est.cutoff        est.quasiMC     est.monitor     est.muTau      ...
              est.lowerbound    est.windowSize};
[isotropic,S,batchsize,lrate,initialize,ncore,epoch,tau,patience,...
 network,dist,seed,c,bvar,Xval,yval,nval,icept,momentum,verbose,cutoff,...
 quasiMC,monitor,muTau,lowerbound,windowSize] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

% Check errors for additional options
% If distribution is 'binomial' but responses are not binary/logical value
if (~isBinomial(y) && strcmp(dist,'binomial'))
    error(deepGLMmsg('deepglm:ResponseMustBeBinary'));
end
% If response is binary array but distribution option is not 'binomial'
if (isBinomial(y) && ~strcmp(dist,'binomial'))
    error(deepGLMmsg('deepglm:DistributionMustBeBinomial'));
end

%% Prepare data and parameters for training phase
% If lowerbound option is true -> do not need validation data
if(lowerbound)
    Xval = [];
    yval = [];
else
    n = size(X,1);                % Number of observation in input dataset
    if (nval <= 1)
        numVal = round(nval*n);       % Number of observation in validation set
    else
        numVal = nval;
    end
    numTrain = n - numVal ;       % Number of observation in training
    % Extract validation set from training data if validation data are not specified
    if(~isnan(seed)) % Set random seed if specified
        rng(seed);
    end
    if(isempty(Xval)&&isempty(yval))
        idxVal = randperm(n,numVal);     % Random indexes of validation data
        Xval = X(idxVal,:);              % Extract subset from input data for validation
        yval = y(idxVal);
        X(idxVal,:) = [];                % Training data
        y(idxVal)   = [];  
    end
end

if(icept)                        % If intercept option is turned on
    X = [ones(size(X,1),1) X];    % Add column of 1 to data
%     Xval = [ones(size(Xval,1),1) Xval];
end  

%% Calculate batchsize in stochastic gradient ascend
if(batchsize<=1)              % If specified batchsize is a propotion
    batchsize = batchsize * size(X,1);
end

if(batchsize>=size(X,1))
    batchsize = size(X,1);
end

if(batchsize > size(X,1))
    error('Batch size must be an positive integer smaller than number of observations in training data');
end

%% Store training settings
est.isIsotropic = isotropic;
est.S = S;
est.batchsize = batchsize;
est.lrate = lrate;
est.initialize = initialize;
est.ncore = ncore;
est.epoch = epoch;
est.tau = tau;
est.patience = patience;
est.network = floor(network);
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

% Check if inputs are valid
checkInput(est);
%% Run different models based on different types of distribution families

if monitor          % If user want to look at training progress
    est = DeepGLMTrainMonitor(X,y,est);
    
else                % Run training using Matlab scripts
    tic
    est = deepGLMTrain(X,y,est);
    CPU = toc;
    disp(['Training time: ',num2str(CPU),'s']);
    est.out.CPU = CPU;      % Save training time
end

end

