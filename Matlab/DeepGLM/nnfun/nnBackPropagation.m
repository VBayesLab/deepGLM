function [gradient,nnOut] = nnBackPropagation(X,y,W_seq,beta,distr)
%NNBACKPROPAGATION Compute gradient of weights in a neural net using 
% backpropagation algorithm
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

n_train = size(X,1);
L = length(W_seq);
a_seq = cell(1,L);
Z_seq = cell(1,L);

a_seq{1} = W_seq{1}*X';
Z_seq{1} = [ones(1,n_train);nnActivation(a_seq{1},'ReLU')];
for j=2:L
    a_seq{j} = W_seq{j}*Z_seq{j-1};
    Z_seq{j} = [ones(1,n_train);nnActivation(a_seq{j},'ReLU')];
end
delta_seq = cell(1,L+1);

% Calculate error at the output layers according to distribution family of
% response
nnOut = beta'*Z_seq{L};
switch distr 
    case 'normal'  
        delta_seq{L+1} = y' - nnOut;
    case 'binomial'
        p_i = 1./(1+exp(-nnOut));
        delta_seq{L+1} = y' - p_i;
    case 'poisson'
        delta_seq{L+1} = y' - exp(nnOut);
end
delta_seq{L} = (beta(2:end)*delta_seq{L+1}).*nnActivationGrad(a_seq{L},'ReLU');
for j=L-1:-1:1
    Wj_tilde = W_seq{j+1};
    Wj_tilde = Wj_tilde(:,2:end);
    delta_seq{j} = (nnActivationGrad(a_seq{j},'ReLU')).*(Wj_tilde'*delta_seq{j+1});
end
gradient_W1 = delta_seq{1}*X;
gradient = gradient_W1(:);
for j = 2:L
    gradient_Wj = delta_seq{j}*(Z_seq{j-1})';
    gradient = [gradient;gradient_Wj(:)];
end
gradient = [gradient;Z_seq{L}*delta_seq{L+1}'];
end

