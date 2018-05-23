% Examples demonstate how to use deepGLM function to fit data with Poisson 
% dependent variable
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%
%   https://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

clear
clc

% load data
% load('../Data/dataBikeSharingPoisson.mat')
% load('../Data/OnlineDeepGLMPoisson.mat')
load('../Data/abaloneDeepGLM.mat')


%% Fit deepGLM model using default setting
% By default, 20% of observations of training data are used for validation
% set. User can change this ratio by option 'nval'
% The default neuron net structure is [n,10,10,1] where n is the
% number of covariates
nn = [2,2,2];
% deepglm1 = deepGLM(X,y,'binomial','nval',0.15,'network',nn);

% User can specify validation data to deepGLM by using 'Xval' and 'yval'
% options. 
mdl = deepGLMfit(X,y,...
              'Distribution','poisson',...
              'Network',nn,...
              'Lrate',0.001,...
              'Seed',1000,...
              'BatchSize',3550,...
              'Verbose',10,...
              'MaxEpoch',10000);

%% Plot training output    
% Plot lowerbound
figure
plot(mdl.out.lbBar,'LineWidth',2)
title('Lowerbound of Variational Approximation','FontSize',20)
xlabel('Iterations','FontSize',14,'FontWeight','bold')
ylabel('Lowerbound','FontSize',14,'FontWeight','bold')
grid on

% Plot shrinkage coefficients
figure
deepGLMplot('Shrinkage',mdl.out.shrinkage,...
            'Title','Shrinkage Coefficients',...
            'Xlabel','Iterations',...
            'LineWidth',2);


%% Prediction on test data
% Make prediction (point estimation) on a test set
Pred1 = deepGLMpredict(mdl,X_test);

% If ytest is specified (for model evaluation purpose)
% then we can check PPS and MSE on test set
Pred2 = deepGLMpredict(mdl,X_test,'ytest',y_test);
disp(['PPS on test data: ',num2str(Pred2.pps)])
disp(['Mean Square Error on test data: ',num2str(Pred2.mse)])

%% Compare with GLM Poisson
X = [X];
y = [y];
X_test = [ones(size(X_test,1),1) X_test];
mdlGLM = glmfit(X,y,'poisson');
y_pred = exp(X_test*mdlGLM);
ppsGLM = mean(-y_test'*X_test*mdlGLM + sum(y_pred));
mseGLM = mean((y_test-y_pred).^2);


