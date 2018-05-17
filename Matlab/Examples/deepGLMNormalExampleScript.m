% Examples demonstate how to use deepGLM function to fit data with continuos 
% dependent variable
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

clear
clc

% load data
% load('../Data/dataSimulationContinuous.mat')
% load('../Data/dataSimulationContinuousEasy.mat')
load('../Data/DirectMarketing.mat')
% load('../Data/SchoolingDataBART.mat')
% load('../Data/SchoolingDataDeepGLM.mat')
% load('../Data/OnlineBART.mat')
% load('../Data/HILDABart.mat')
% load('../Data/abaloneBART.mat')

%% Fit deepGLM model using default setting
% By default, if 'distribution' option is not specified then deepGLMfit
% will assign the response variables as 'normal'
nn = [5,5];
mdl = deepGLMfit(X,y,...  
                 'Distribution','normal',...
                 'Network',nn,... 
                 'Lrate',0.01,...           
                 'Verbose',10,...             % Display training result each iteration
                 'BatchSize',size(X,1),...   % Use entire training data as mini-batch
                 'MaxEpoch',10000,...
                 'Patience',50,...           % Higher patience values could lead to overfitting
                 'Seed',100,...
                 'Intercept',true);
             
%% Plot training output
figure
plot(mdl.out.lbBar,'LineWidth',2)
title('Lowerbound of Variational Approximation','FontSize',0.5)
xlabel('Iterations','FontSize',0.2,'FontWeight','bold')
ylabel('Lowerbound','FontSize',0.2,'FontWeight','bold')
grid on

%% Plot shrinkage coefficients
figure
deepGLMplot('Shrinkage',mdl.out.shrinkage,...
            'Title','Shrinkage Coefficients',...
            'Xlabel','Iterations',...
            'LineWidth',2);

%% Prediction on test data
% Make prediction (point estimation) on a test set
disp('---------- Prediction ----------')
Pred1 = deepGLMpredict(mdl,X_test);

% If ytest is specified (for model evaluation purpose)
% then we can check PPS and MSE on test set
Pred2 = deepGLMpredict(mdl,X_test,'ytest',y_test);
disp(['PPS on test set using deepGLM is: ',num2str(Pred2.pps)])
disp(['MSE on test set using deepGLM is: ',num2str(Pred2.mse)])

% You can also perform point and interval estimation for a single test observation
idx = randi(length(y_test));     % Pick a random test data observation
dataTest = X_test(idx,:);
Pred3 = deepGLMpredict(mdl,dataTest,...
                       'Interval',1,...
                       'Nsample',1000);
disp(['Prediction Interval: [',num2str(Pred3.interval(1)),...
                                     ';',num2str(Pred3.interval(2)),']',]);
disp(['True value: ',num2str(y_test(idx))]);
  

% Estimate prediction interval for entire test data
Pred4 = deepGLMpredict(mdl,X_test,...
                      'ytest',y_test,...
                      'Interval',1,...
                      'Nsample',1000);                       
y_pred = mean(Pred4.yhatMatrix)';
mse2 = mean((y_test-y_pred).^2);
accuracy = (y_test<Pred4.interval(:,2) & y_test>Pred4.interval(:,1));
disp(['Prediction Interval accuracy: ',num2str(sum(accuracy)/length(accuracy))]);

%% Plot prediction interval
figure
deepGLMplot('Interval',Pred4,...
            'Title','Prediction Interval of Schooling Test Data',...
            'Xlabel','Observations',...
            'Ylabel','Wage($1000)',...
            'Nsample',60);
        
%% Plot prediction interval with true response
figure
deepGLMplot('Interval',Pred4,...
            'ytest',y_test,...
            'Title','Prediction Interval for Test Data',...
            'Xlabel','Observations',...
            'Ylabel','Wage($1000)',...
            'Nsample',40);
                              
