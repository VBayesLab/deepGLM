function deepGLMplot(type,Pred,varargin)
%DEEPGLMPLOT Plot analytical figures for deepGLM estimation and prediction
%
%   DEEPGLMPLOT(TYPE, DATA) Plots data specified in data1 according the type 
%   specified by TYPE. TYPE can be one of the following options:
%
%      'Shrinkage'        Plot stored values of group Lasso coefficient during 
%                         training phase. If this type is specify, DATA is the 
%                         output structure mdl from DEEPGLMFIT or user can manually 
%                         extract mdl.out.shrinkage field from mdl an use as input 
%                         argument data.
%      'Interval'         Plot prediction interval estimation for test data. 
%                         If this type is specified, DATA is the output structure 
%                         Pred from DEEPGLMPREDICT.  
%      'ROC'              Plot ROC curve for prediction from binary test data. 
%                         If this type is specify, data is a matrix where the 
%                         first column is a vector of target responses and the 
%                         second column is the predicted vector yProb extract 
%                         from output structure Pred of deepGLMpredict. If you 
%                         want to plot different ROC curves, add more probability 
%                         columns to data.  
%
%   DEEPGLMPLOT(TYPE, DATA, NAME, VALUE) Plots data specified in DATA according 
%   the type specified by type with one of the following NAME/VALUE pairs:
%
%      'Nsample'          This option only available when ‘type’ is Interval. 
%                         'Nsample' specifies number of test observations 
%                         randomly selected from test data to plot prediction 
%                         intervals. 
%      'Title'            Title of the figure   
%      'Xlabel'           Label of X axis  
%      'Ylabel'           Label of Y axis 
%      'LineWidth'        Line width of the plots
%      'Legend'           Legend creates a legend with descriptive labels for 
%                         each plotted data series
%
%   Example: See deepGLMNormalExample.mlx and deepGLMBinomialExample.mlx
%   in the Examples folder
%
%   See also DEEPGLMFIT, DEEPGLMPREDICT
%
%   Copyright 2018:
%       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
%      
%   https://github.com/VBayesLab/deepGLM
%
%   Version: 1.0
%   LAST UPDATE: May, 2018

% Check errors input arguments
if nargin < 2
    error(deepGLMmsg('deepglm:TooFewInputs'));
end

%% Parse additional options
paramNames = {'Title'          'Xlabel'          'Ylabel'         'LineWidth',...
              'Color'          'IntervalStyle'   'Nsample'        'Ordering',...
              'ytest'          'Legend'};
          
paramDflts = {''               ''                ''               2,...                 
              'red'            'shade'           50              'ascend',...
              []               {}};
          
[TextTitle,labelX,labelY,linewidth,color,style,npoint,order,y,legendText] =...
                internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
% Store plot options to a structure
opt.title = TextTitle;
opt.labelX = labelX;
opt.labelY = labelY;
opt.linewidth = linewidth;
opt.color = color;
                                            
switch type
    case 'Shrinkage'
        plotShrinkage(Pred,opt);
    case 'Interval'
        yhat = Pred.yhatMatrix;
        yhatInterval = Pred.interval;
        predMean = mean(yhat);
        % If test data have more than 100 rows, extract randomly 100 points to draw
        if(length(predMean)>=npoint)
            idx = randperm(length(yhatInterval),npoint);
            intervalPlot = yhatInterval(idx,:);
            yhatMeanPlot = predMean(idx)';
            if(~isempty(y))
                 ytruePlot = y(idx)';
            end
        else
            yhatMeanPlot = predMean';
            intervalPlot = yhatInterval;
            ytruePlot = y;
        end
        % Sort data
        [yhatMeanPlot,sortIdx] = sort(yhatMeanPlot,order);
        intervalPlot = intervalPlot(sortIdx,:);
        if(isempty(y))
            ytruePlot = [];
        else
            ytruePlot = ytruePlot(sortIdx);
        end
        plotInterval(yhatMeanPlot,intervalPlot,opt,...
                    'ytrue',ytruePlot,...
                    'Style',style);
    case 'ROC'
        if(~isnumeric(y))
            disp('Target should be a column of binary responses!')
            return
        else
            % Plot single ROC
            if(size(Pred,2)==1)
                [tpr,fpr,~] = roc(y',Pred');
                plot(fpr,tpr,'LineWidth',linewidth);
                grid on
                title(TextTitle,'FontSize',20);
                xlabel(labelX,'FontSize',15);
                ylabel(labelY,'FontSize',15);
            % Plot multiple ROC
            else
                tpr = cell(1,size(Pred,2));
                fpr = cell(1,size(Pred,2));
                for i=1:size(Pred,2)
                    [tpr{i},fpr{i},~] = roc(y',Pred(:,i)');
                    plot(fpr{i},tpr{i},'LineWidth',linewidth);
                    grid on
                    hold on
                end
                title(TextTitle,'FontSize',20);
                xlabel(labelX,'FontSize',15);
                ylabel(labelY,'FontSize',15);
                legend(legendText{1},legendText{2});
            end
        end
end

end

% plot(fpr,tpr,'r',fpr1,tpr1,'--g','LineWidth',3)
% legend({'deepGLM','BART'},'FontSize',16) 
% grid on
% title('ROC: deepGLM vs BART',...
%                                   'FontSize',22,'FontWeight','bold')
% xlabel('False Positive Rate','FontSize',16, 'FontWeight','bold')
% ylabel('True Positive Rate','FontSize',16, 'FontWeight','bold')
%                               
