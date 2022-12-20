function out = nnActivationGrad(z,func)
%NNACTIVATIONGRAD Calculate derivative of activation output at hidden nodes 
%in each backward pass
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018


switch func
    case 'Linear'
        out = ones(size(z));
    case 'Sigmoid'
        temp = activation(z,text);
        out = temp.*(1-temp);
    case 'Tanh'
        temp = activation(z,text);
        out = 1 - temp^2;
    case 'ReLU'
        out = z>0;
    case 'LeakyReLU'
        if z > 0
            out = 1;
        else 
            out = alpha;
        end
end

end

