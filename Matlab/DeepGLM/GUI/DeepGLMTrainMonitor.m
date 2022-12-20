function varargout = DeepGLMTrainMonitor(varargin)
% DEEPGLMTRAINMONITOR MATLAB code for DEEPGLMTRAINMONITOR.fig
%      DEEPGLMTRAINMONITOR, by itself, creates a new DEEPGLMTRAINMONITOR or raises the existing
%      singleton*.
%
%      H = DEEPGLMTRAINMONITOR returns the handle to a new DEEPGLMTRAINMONITOR or the handle to
%      the existing singleton*.
%
%      DEEPGLMTRAINMONITOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DEEPGLMTRAINMONITOR.M with the given input arguments.
%
%      DEEPGLMTRAINMONITOR('Property','Value',...) creates a new DEEPGLMTRAINMONITOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DEEPGLMTRAINMONITOR gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DEEPGLMTRAINMONITOR_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DEEPGLMTRAINMONITOR

% Last Modified by GUIDE v2.5 10-May-2018 13:17:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DeepGLMTrainMonitor_OpeningFcn, ...
                   'gui_OutputFcn',  @DeepGLMTrainMonitor_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before DeepGLMTrainMonitor is made visible.
function DeepGLMTrainMonitor_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DeepGLMTrainMonitor (see VARARGIN)

global est stopFlag X y lbGlobal LassoGlobal

lbGlobal = [];
LassoGlobal=[];

% Choose default command line output for DeepGLMTrainMonitor
handles.output = hObject;

set(gcf, 'units', 'normalized', 'position', [0.05 0.05 0.9 0.9])
% Get input data from caller.
% est = getappdata(0,'Settings');
X = varargin{1};
y = varargin{2};
est = varargin{3};
stopFlag = false;

% Display input setting
% Display network structure
networkText = '';
for i=1:length(est.network)
    if(~isempty(networkText))
        networkText = [networkText,',',num2str(est.network(i))];
    else
        networkText = [networkText,num2str(est.network(i))];
    end
end
set(handles.txtNetwork,'string',networkText);
set(handles.txtlrate,'string',num2str(est.lrate));         % Display learning rate
set(handles.txtbatchsize,'string',num2str(est.batchsize)); % Display batch size
set(handles.txtTau,'string',num2str(est.tau));             % Display tau
set(handles.txtPatience,'string',num2str(est.patience));   % Display patience
set(handles.txtMomentum,'string',num2str(est.momentum));   % Display momentum
set(handles.txtCutoff,'string',num2str(est.cutoff));       % Display cutoff probability
set(handles.txtS,'string',num2str(est.S));                 % Display S
set(handles.txtC,'string',num2str(est.c));                 % Display C
set(handles.txtB,'string',num2str(est.bvar));              % Display b
set(handles.txtDistr,'string',est.dist);  
set(handles.txtSeed,'string',est.seed);                    % Display random seed
set(handles.txtInit,'string',est.initialize); 
if(est.isIsotropic)
    set(handles.txtIsotropic,'string','Yes');
else
    set(handles.txtIsotropic,'string','No');
end
set(handles.txtCovariate,'string',num2str(size(X,2))); 
set(handles.txtTrainData,'string',num2str(length(X)));
% set(handles.etVerbose,'string',num2str(est.verbose));  
set(handles.txtNumEpoch,'string',num2str(est.epoch));
time = datestr(datetime('now'));
set(handles.txtstart_time,'string',time);
niter = round((length(est.data.X)/est.batchsize));
set(handles.txtIterEpoch,'string',num2str(niter));
set(handles.btnClose,'Enable','off')
est.startTime = datetime('now');

axes(handles.axes_pps)
xlabel('Iteration','FontWeight','bold')
ylabel('Lowerbound','FontWeight','bold')

% Create a timer, but do not start it
handles.timer = timer(...
    'ExecutionMode', 'fixedRate', ...
    'Period', .5, ...
    'TimerFcn', @(obj, event)update_plot(handles));
start(handles.timer);



guidata(hObject, handles);

% UIWAIT makes DeepGLMTrainMonitor wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DeepGLMTrainMonitor_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
% global est
% 
% handles.output = est;

global est X y
% Run DeepGLM training
tic
est = deepGLMTrainGUI(X,y,est,handles);
est.out.CPU = toc;
set(handles.btnClose,'Enable','on')
if(isvalid(handles.timer))
    stop(handles.timer);
    delete(handles.timer);
end

% Change text to save output
set(handles.btnStopTrain,'String','Save Output')

% Display CPU
set(handles.txtTime,'string',num2str(est.out.CPU));

% Plot shrinkage in a separated windows
figure
deepGLMplot('Shrinkage',est.out.shrinkage,...
            'Title','Shrinkage Coefficients',...
            'Xlabel','Iterations',...
            'LineWidth',2);
 
% Make prediction if test data is provided
if(size(est.data.Xtest,2)>1)
    Pred2 = deepGLMpredict(est,est.data.Xtest,'ytest',est.data.ytest);
    set(handles.txtPPS,'string',num2str(Pred2.pps));
    set(handles.txtMSE,'string',num2str(Pred2.mse));
end

% Save result to a directory 
        
% varargout{1} = handles.output;
varargout{1} = est;



% --- Executes on button press in btnClose.
function btnClose_Callback(hObject, eventdata, handles)
% hObject    handle to btnClose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global stopFlag

stopFlag = true;

% Stop the timer
if(isvalid(handles.timer))
    stop(handles.timer);
    delete(handles.timer);
end

selection = questdlg('Close the Program?',...
  'Exit',...
  'Yes','No','Yes'); 
switch selection 
  case 'Yes'
     close DeepGLMTrainMonitor force
  case 'No'
  return 
end


% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in btnStopTrain.
function btnStopTrain_Callback(hObject, eventdata, handles)
% hObject    handle to btnStopTrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global stopFlag est
clear text
% Stop the timer
if(isvalid(handles.timer))
    stop(handles.timer);
    delete(handles.timer);
end
set(handles.btnClose,'Enable','on')

text = get(handles.btnStopTrain,'String');
if(strcmp(text,'Save Output'))
    filter = {'*.mat'};
    [file, path] = uiputfile(filter);
    if(length(file)>1 && length(path)>1)
        save([path,file],'est')
    end
end

% Change text to save output
set(handles.btnStopTrain,'String','Save Output')

% Stop training 
stopFlag = true;

function update_plot(handles)
% This function updates the plot
global lbGlobal LassoGlobal
% handles = guidata(hObject);
if(~isempty(lbGlobal))
    axes(handles.axes_pps)
    plot(handles.axes_pps, lbGlobal,'LineWidth',2);
    grid on
    xlabel('Iteration','FontWeight','bold')
    ylabel('Lowerbound','FontWeight','bold')
end

% axes(handles.axesLasso)
% plot(handles.axesLasso,LassoGlobal','LineWidth',linewidth);
% grid on
% xlabel(Iteration,'FontWeight','bold')
% Ytext = lbGlobal(:,end);  % Y coordination of text, different for coefficients
% Xtext = size(lbGlobal,2); % X coordination of text, same for all coefficients 
% for i=1:numCoeff
%     text(Xtext,Ytext(i),['\gamma_{',num2str(i),'}'])
% end
