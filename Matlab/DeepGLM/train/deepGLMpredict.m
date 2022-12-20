function out = deepGLMpredict(mdl,X,varargin)
%DEEPGLMPREDICT Make prediction from a trained deepGLM model
%
%   OUT = DEEPGLMPREDICT(MDL,XTEST) predict responses for new data XTEST using 
%   trained deepGLM structure MDL (output from DEEPGLMFIT) 
%
%   OUT = DEEPGLMPREDICT(MDL,XTEST,NAME,VALUE) predicts responses with additional 
%   options specified by one or more of the following name/value pairs:
%
%      'ytest'            Specify column of test responses. If this option 
%                         is specified with true response column of new 
%                         observations, deepGLMpredict will return prediction 
%                         scores (PPS, MSE or Classification Rate) using true 
%                         responses column vector ytest
%      'Interval'         Return prediction interval estimation for observations 
%                         in test data Xtest. By default, this predictive 
%                         interval capability is disable ('Interval' is 0). 
%                         Must be an positive number.   
%      'Nsample'          Number of samples generated from posterior distribution 
%                         of model parameters used to make prediction interval 
%                         estimation for test data. Must be a positive integer
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
%      Pred = deepGLMpredict(mdl,X_test,...
%                           'ytest',y_test);
%      disp(['PPS on test data: ',num2str(Pred.pps)])
%      disp(['MSE on test data: ',num2str(Pred.mse)])
%   
%   For more examples, check EXAMPLES folder
%
%   See also DEEPGLMFIT, DEEPGLMPLOT
%
%   Copyright 2018:
%       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
%      
%   https://github.com/VBayesLab/deepGLM
%
%   Version: 1.0
%   LAST UPDATE: May, 2018

% Check errors input arguments
if nargin < 2
    error(deepGLMmsg('deepglm:TooFewInputs'));
end

% Load deepGLM params from struct
W_seq = mdl.out.weights;
beta = mdl.out.beta;
distr = mdl.dist;

% Parse additional options
paramNames = {'ytest'      'Interval'      'Nsample'       'Intercept'};
paramDflts = {[]           0               1000            true};
[y,alpha,Nsample,intercept] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

% If y test is specified, check input
if(~isempty(y))
    if size(y,1) ~= size(X,1)
        error(deepGLMmsg('deepglm:InputSizeMismatchX'));
    end
    if size(y,2) ~= 1
        error(deepGLMmsg('deepglm:InputSizeMismatchY'));
    end
end

% Add column of 1 to X if intercept is true
if(intercept)
    X = [ones(size(X,1),1),X];
end

% Store Nsample to deepGLMfit
mdl.Nsample = Nsample;

% Calculate neuron network output
nnet_output = nnFeedForward(X,W_seq,beta);

switch distr
    case 'normal'
        out.yhat = nnet_output;    % Prediction for continuous response
        % If ytest if provided, then calculate pps and mse
        if(~isempty(y))
            sigma2 = mdl.out.sigma2Mean;
            mse = mean((y-nnet_output).^2);
            pps = 1/2*log(sigma2) + 1/2/sigma2*mse;
            out.mse = mse;
            out.pps = pps;
        end
        % Calculate confidence interval if required
        if(alpha~=0)
            interval = predictionInterval(mdl,X,alpha);
            out.interval = interval.interval;
            out.yhatMatrix = interval.yhatMC;
        end
        
    case 'binomial'
        out.yNN = nnet_output;
        out.yProb = exp(nnet_output)./(1+exp(nnet_output));
        y_pred = double(nnet_output>0);   % Prediction for binary response
        out.yhat = y_pred;
        % If ytest if provided, then calculate pps and mse
        if(~isempty(y))
            pps = mean(-y.*nnet_output+log(1+exp(nnet_output)));
            cr = mean(y==y_pred);    % Miss-classification rate
            out.pps = pps;
            out.accuracy = cr;
        end
        
    case 'poisson'
        out.yNN = nnet_output;
        y_pred = exp(nnet_output);        % Prediction for poisson response
        out.yhat = y_pred;
        if(~isempty(y))
            pps = mean(-y.*nnet_output+exp(nnet_output));
            mse = mean((y-y_pred).^2);
            out.mse = mse;
            out.pps = pps;
        end
end
end

