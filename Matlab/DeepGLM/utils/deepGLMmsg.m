function msg_out = deepGLMmsg(identifier)
%DEEPGLMMSG Define custom error/warning messages for exceptions
%   DEEPGLMMSG = (IDENTIFIER) extract message for input indentifier
%   
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

switch identifier
    case 'deepglm:TooFewInputs'
        msg_out = 'At least two arguments are specified';
    case 'deepglm:InputSizeMismatchX'
        msg_out = 'X and Y must have the same number of observations';
    case 'deepglm:InputSizeMismatchY'
        msg_out = 'Y must be a single column vector';
    case 'deepglm:ArgumentMustBePair'
        msg_out = 'Optinal arguments must be pairs';
    case 'deepglm:ResponseMustBeBinary'
        msg_out = 'Two level categorical variable required';
    case 'deepglm:DistributionMustBeBinomial'
        msg_out = 'Binomial distribution option required';
    case 'deepglm:MustSpecifyActivationFunction'
        msg_out = 'Activation function type requied';
end
end

