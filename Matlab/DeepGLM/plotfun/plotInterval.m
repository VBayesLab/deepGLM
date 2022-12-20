function plotInterval(predMean,predInterval,opt,varargin)
%PLOTINTERVAL Plot prediction interval for test data
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

if (nargin<2)
    disp('ERROR: not enough input arguments!');
    return;
end

textTitle = opt.title;
labelX = opt.labelX;
labelY = opt.labelY;
linewidth = opt.linewidth;

% Define some default texts
if(isempty(textTitle))
    textTitle = 'Prediction Interval on Test Data';
end
if(isempty(labelX))
    labelX = 'Observation';
end

% Parse additional options
paramNames = {'Color'     'Style'       'ytrue'};
paramDflts = {'red'       'shade'       []};
[color,style,ytrue] = internal.stats.parseArgs(paramNames,...
                                                 paramDflts, varargin{:});
                          
lower = predInterval(:,1);
upper = predInterval(:,2);
t = 1:1:length(predMean);
switch style
    case 'shade'       % Plot prediction interval in shade style
        p = plot(t,predMean,t,upper,t,lower);
        YLIM = get(gca,'YLim');    
        delete(p);
        a1 = area(t,upper,min(YLIM)); 
        hold on;
        set(a1,'LineStyle','none');     
        set(a1,'FaceColor',[0.9 0.9 0.9]);
        a2 = area(t,lower,min(YLIM)); 
        set(a2,'LineStyle','none');     
        set(a2,'FaceColor',[1 1 1]);
        p2 = scatter(t,predMean,40,'MarkerEdgeColor',[1 0 0]);
        if(~isempty(ytrue))
            p1 = scatter(t,ytrue,40,'MarkerEdgeColor',[0 0 1]);
            legend([p1,p2],{'True values','Prediction values'});
        end
        title(textTitle, 'FontSize',18)
        xlabel(labelX)
        ylabel(labelY)
        hold off;           
        set(gca,'Layer','top','XGrid','on','YGrid','on');
    case 'boundary'   % Plot prediction interval in boundary style 
        plot(t,predMean,'LineWidth',linewidth,'Color',color);
        hold on
        plot(t,upper,'--r',t,lower,'--r');
        grid on
        title('Prediction Interval on Test Data', 'FontSize',18)
        xlabel('Observation')
        hold off
    case 'bar'        % Plot prediction interval in bar style 
        err = (upper-lower)/2;
        errorbar(predMean,err);
        grid on
        hold on
        plot(predMean,'Color','red','LineWidth',2);
        title('Prediction Interval on Test Data', 'FontSize',18)
        xlabel('Observation')
        hold off
end
end

