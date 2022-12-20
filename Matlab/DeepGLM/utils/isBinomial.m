function out = isBinomial(array)
%ISBINOMIAL Check if an array are binary vector

%   Copyright 2018 
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

out = false;
uniqueVal = unique(array);    % Extract unique values in array
if (length(uniqueVal)==2) && (uniqueVal(1)==0) && (uniqueVal(2)==1)
    out = true;
end
end

