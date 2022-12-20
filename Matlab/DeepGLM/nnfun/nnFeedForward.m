function nnOutput = nnFeedForward(X,W_seq,beta)
%NNFEEDFORWARD Compute the output of a neural net 

%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

% Number of observations in dataset
n_train = size(X,1); 

% Make forward passes to all layers
a = W_seq{1}*X';
Z = [ones(1,n_train);nnActivation(a,'ReLU')];
L = length(W_seq);
for j=2:L
    a = W_seq{j}*Z;
    Z = [ones(1,n_train);nnActivation(a,'ReLU')]; % Add biases 
end
% a = W_seq{L}*Z;
% Z = [ones(1,n_train);nnActivation(a,'ReLU')];
nnOutput = Z'*beta;
end

