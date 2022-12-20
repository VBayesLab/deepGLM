function checkInput(est)
%CHECKDATA Check if user input correct model settings
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au).
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018
dist = est.dist;
network = est.network;
lrate = est.lrate;
momentum = est.momentum;
batchsize = est.batchsize;
epoch = est.epoch;
patience = est.patience;
tau = est.tau;
S = est.S;
windowSize = est.windowSize;
icept = est.icept;
verbose = est.verbose;
monitor = est.monitor;
isotropic = est.isIsotropic;
seed = est.seed;

if(~strcmpi(dist,'normal') && ~strcmpi(dist,'binomial') && ~strcmpi(dist,'poisson'))
    error(['Distribution must be one of the followings: ','normal,','binomial,','poisson']);
end

if(sum(network==0)>0)
    error('Network must be an array of positive integers')
end

if(sum(network==0)>0)
    error('Network must be an array of positive integers')
end

if(~isnumeric(lrate) || lrate<=0)
    error('Learning rate must be a positive numerical value')
end

if(~isnumeric(momentum) || momentum<0 || momentum > 1)
    error('Momentum must be a numerical value from 0 to 1')
end

if(~isnumeric(batchsize) || floor(batchsize)~= batchsize || batchsize <= 0)
    error('Batch size must be an positive integer smaller than number of observations in training data');
end

if(~isnumeric(epoch) || floor(epoch)~= epoch || epoch <= 0)
    error('Number of epoches must be a positive integer');
end

if(~isnumeric(patience) || floor(patience)~= patience || patience <= 0)
    error('Patience must be a positive integer');
end

if(~isnumeric(tau) || floor(tau)~= tau || tau <= 0)
    error('LrateFactor must be a positive integer');
end

if(~isnumeric(S) || floor(S)~= S || S <= 0)
    error('S must be a positive integer');
end

if(~isnumeric(windowSize) || floor(windowSize)~= windowSize || windowSize <= 0)
    error('WindowSize must be a positive integer');
end

if(~islogical(icept))
    error('Intercept option must be a logical value');
end

if(~isnumeric(verbose) || floor(verbose)~= verbose || verbose <= 0)
    error('Verbose must be a positive integer');
end

if(~islogical(monitor))
    error('Monitor option must be a logical value');
end

if(~islogical(isotropic))
    error('Isotropic option must be a logical value');
end

if (~isnan(seed))
    if(~isnumeric(seed) || floor(seed)~= seed || seed <= 0)
        error('Seed must be a nonnegative integer less than 2^32');
    end
end
end

