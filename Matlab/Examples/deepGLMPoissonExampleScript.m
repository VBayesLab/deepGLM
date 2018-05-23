% Examples demonstate how to use deepGLM function to fit data with Poisson 
% dependent variable
%
% Copyright 2018 
%                Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%                Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) 
%
% https://github.com/VBayesLab/deepGLM
% 
% Version: 1.0
% LAST UPDATE: May, 2018

clear
clc

clear
clc

% load data
% load('../Data/dataBikeSharingPoisson.mat')
load('../Data/abaloneBART.mat')


%% Fit deepGLM model using default setting
nn = [10,10];
mdl = deepGLMfit(X,y,...
              'Distribution','poisson',...
              'Network',nn,...
              'Lrate',0.005,...
              'BatchSize',size(X,1),...
              'MaxEpoch',1000,...
              'Patience',50,...
              'Verbose',10,...
              'Seed',1000);

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
mdlGLM = glmfit(X,y,'poisson');
X_test = [ones(size(X_test,1),1) X_test];
y_pred = exp(X_test*mdlGLM);
ppsGLM = mean(-y_test'*X_test*mdlGLM + sum(y_pred));
mseGLM = mean((y_test-y_pred).^2);


