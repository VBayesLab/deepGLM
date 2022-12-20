function plotROC(y_true,y_pred)
%PLOTROC Plot ROC curve and AUC

if nargin<2
    disp('Too few input arguments');
    return
end

if(size(y_true)~=size(y_pred))
    disp('Target and output must have same size')
    return
elseif(size(y_true,1)~=1)
    disp('Target and output must be row vectors with same length')
    return
else
    plotroc(y_true,y_pred)
    grid on
end

end

