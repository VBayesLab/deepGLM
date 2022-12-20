function predInterval = predictionInterval(mdl,X,zalpha)
%CONFIDENTINTERVAL Interval estimation for test data using deepGLM
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

% Load deepGLM params from struct
Nsample = mdl.Nsample;
MU = mdl.out.vbMU;
SIGMA = mdl.out.vbSIGMA;
n_units = mdl.network;
index_track = mdl.out.indexTrack;
alpha_sigma2 = mdl.out.sigma2Alpha;
beta_sigma2 = mdl.out.sigma2Beta;

% Calculate network parameters 
L = length(n_units);        % Number of hidden layers
p = size(X,2)-1;            % Number of covariates
d_beta = n_units(L)+1; 
d_w = index_track(L);

yhat = zeros(Nsample,size(X,1));      % Predicted values of test data
nnOut = zeros(Nsample,size(X,1));     % Output of NN
for i=1:Nsample
    % Generate samples of theta from Normal distribution
    theta_i = mvnrnd(MU,SIGMA);   
    % Generate samples of sigma from IG distribution
    sigma2_i = 1/gamrnd(alpha_sigma2,1./beta_sigma2);
    
    % For each generated theta, restore neuron net structure
    W1 = reshape(theta_i(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1; 
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(theta_i(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
    end
    beta = theta_i(d_w+1:d_w+d_beta)';
    
    % Calculate neuron network output
    nnOut(i,:) = nnFeedForward(X,W_seq,beta);
    
    % Calculate p(y|theta_i,sigma_i,X)
    yhat(i,:) = normrnd(nnOut(i,:),sqrt(sigma2_i));
    
end

% 95% confidence interval
yhatLCL = mean(yhat) - zalpha*std(yhat);
yhatUCL = mean(yhat) + zalpha*std(yhat);
yhatInterval = [yhatLCL',yhatUCL'];
predInterval.yhatMC = yhat;
predInterval.interval = yhatInterval;
end

