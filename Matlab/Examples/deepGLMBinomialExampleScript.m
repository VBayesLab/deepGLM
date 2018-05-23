% Examples demonstate how to use deepGLM function to fit data with binomial 
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

% load data
% load('../Data/dataSimulationBinary.mat')
load('../Data/DataSimulationBinary.mat')

%% Fit deepGLM model using default setting
nn = [10];
lb = true;
mdl = deepGLMfit(X,y,... 
                 'Distribution','binomial',...
                 'Network',nn,... 
                 'Lrate',0.01,...           
                 'Verbose',1,...             % Display training result each iteration
                 'BatchSize',size(X,1),...   % Use entire training data as mini-batch
                 'MaxEpoch',10000,...
                 'Patience',50,...           % Higher patience values could lead to overfitting
                 'Lowerbound',lb,...         % Use lowerbound as early stopping condition and model selection
                 'Seed',100);
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
disp(['Classification rate on test data: ',num2str(Pred2.accuracy)])

% Plot ROC curve
figure
deepGLMplot('ROC',Pred2.yProb,...
            'ytest',y_test,...
            'Title','ROC',...
            'Xlabel','False Positive Rate',...
            'Ylabel','True Positive Rate')

%% Compare to linear model
figure
mdlLR = fitglm(X,y,'Distribution','binomial','Link','logit');
yProb = predict(mdlLR,X_test);
deepGLMplot('ROC',[Pred2.yProb,yProb],...
            'ytest',y_test,...
            'Title','ROC',...
            'Xlabel','False Positive Rate',...
            'Ylabel','True Positive Rate',...
            'legend',{'deepGLM','Logistic Regression'})

