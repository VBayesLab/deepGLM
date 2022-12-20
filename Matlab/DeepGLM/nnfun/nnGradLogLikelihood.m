function [gradient,nnOut] = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse)
%NNGRADIENTLLH Calculate gradient of log likelihood
%   Detailed explanation goes here
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

n = length(y);
[back_prop,nnOut] = nnBackPropagation(X,y,W_seq,beta,distr);
nnOut = nnOut';
switch distr
    case 'normal'
        gradient_theta = mean_sigma2_inverse*back_prop;
        gradient = datasize/n*gradient_theta;   % To compensate the variation
    case 'binomial'
        gradient = datasize/n*back_prop;
    case 'poisson'
        gradient = datasize/n*back_prop;
end
end

