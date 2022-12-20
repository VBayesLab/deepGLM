function plotPPS(loss,data)
%PLOTPPS Plot prediction loss
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

plot(loss);
grid on;
title(['Prediction Loss on ',data,' set']);
xlabel('Iterations');
ylabel('PPS')
end

