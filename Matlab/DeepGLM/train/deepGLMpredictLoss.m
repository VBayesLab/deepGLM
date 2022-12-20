function [out1,out2] = deepGLMpredictLoss(X,y,W_seq,beta,distr,sigma2)
%DEEPGLMPREDICTION Make prediction from estimated deepGLM model
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

% Calculate neuron network output
nnet_output = nnFeedForward(X,W_seq,beta);

switch distr
    case 'normal'
        mse = mean((y-nnet_output).^2);
        pps = 1/2*log(sigma2) + 1/2/sigma2*mse;
        out2 = mse;
    case 'binomial'
        pps = mean(-y.*nnet_output+log(1+exp(nnet_output)));
        y_pred = nnet_output>0;
        mcr = mean(abs(y-y_pred));        % Miss-classification rate
        out2 = 1 - mcr;                   % Report output in classification rate   
    case 'poisson'
        pps = mean(-y.*nnet_output+exp(nnet_output));
        mse = mean((y-exp(nnet_output)).^2);
        out2 = mse;
end
out1 = pps;

end

