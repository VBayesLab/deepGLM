function S = sumResidualSquared(y,X,W_seq,beta)
% compute the sum_residual_squared for normal-NN model

nnet_output = nnFeedForward(X,W_seq,beta);
S = sum((y-nnet_output).^2);
end



