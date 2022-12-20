function out = nnActivation(z,func)
%NNACTIVATION Calculate activation output at nodes in each forward pass

%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

if nargin < 2
    error(deepGLMmsg('deepglm:MustSpecifyActivationFunction'));
end

switch func
    case 'Linear'
        out = z;
    case 'Sigmoid'
        out = 1.0 ./ (1.0 + exp(-z));
    case 'Tanh'
        out = tanh(z);
    case 'ReLU'
        out = max(0,z);
    case 'LeakyReLU'
        out = max(0,z)+ alpha*min(0,z);
end
end

