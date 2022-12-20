function out = nnSumResidualSquare(y,X,W_seq,beta)
%NNSUMRESIDUALSQUARE Calculate sum of square of residuals
%
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au).
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

nnet_output = nnFeedForward(X,W_seq,beta);
out = sum((y-nnet_output).^2);
end

