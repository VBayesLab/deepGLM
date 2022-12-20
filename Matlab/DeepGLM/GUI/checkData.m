function isOK = checkData(X,y)

% Check if number of covariates of train and test data are the same
if(size(X,1)~=size(y,1))
    txt = 'Number of observations for X and y of train and test data should be the same!';
    errordlg(txt,'Data Import Error');
else
    isOK = true;
end
        
end

