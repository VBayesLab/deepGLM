function est = deepGLMnormalCV(X_train,y_train,est)
% Traing a fGLM model with continuous reponse y.
% Bayesian Adaptive Group Lasso is used on the first-layer weights; no
% regularization is put on the rest. sigma2 and tau are updated by
% mean-field VB. Inverse gamma prior is used for sigma2
% INPUT
%   X_train, y_train:           Training data (continuous response)
%   X_validation, y_validation: Validation data
%   n_units:                    Vector specifying the numbers of units in
%                               each layer
%   batchsize:                  Mini-batch size used in each iteration
%   eps0:                       Constant learning rate
%   isotropic:                  True if isotropic structure on Sigma is
%                               used, otherwise rank-1 structure is used
% OUTPUT
%   W_seq:                      The optimal weights upto the last hidden
%                               layer
%   beta                        The optimal weights that connect last hidden layer to the output
%   mean_sigma2                 Estimate of sigma2
%   shrinkage_gamma_seq         Update of shrinkage parameters over
%                               iteration
%   MSE_DL                      Mean squared error over iteration
%
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   http://www.xxx.com
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

% Extract training data and settings from input struct
X_val = est.data.Xval;
y_val = est.data.yval;
n_units = est.network;
batchsize = est.batchsize;
lrate = est.lrate;
isotropic = est.isIsotropic;
S = est.S;                   % Number of Monte Carlo samples to estimate the gradient
tau = est.tau;               % Threshold before reducing constant learning rate eps0
grad_weight = est.momentum;  % Weight in the momentum 
cScale = est.c;              % Random scale factor to initialize b,c
patience = est.patience;     % Stop if test error not improved after patience_parameter iterations
epoch = est.epoch;           % Number of times learning algorithm scan entire training data
verbose = est.verbose;
distr = est.dist;

% Data merge for mini-batch sampling
data = [y_train,X_train];                 
datasize = length(y_train);
num1Epoch = round(datasize/batchsize);    % Number of iterations per epoch

% Network parameters
L = length(n_units);        % Number of hidden layers
p = size(X_train,2)-1;      % Number of covariates
W_seq = cell(1,L);          % Cells to store weight matrices
index_track = zeros(1,L);   % Keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
index_track(1) = n_units(1)*(p+1);            % Size of W1 is m1 x (p+1) with m1 number of units in the 1st hidden layer 
W1_tilde_index = n_units(1)+1:index_track(1); % Index of W1 without biases, as the first column if W1 are biases
for j = 2:L
    index_track(j) = index_track(j-1)+n_units(j)*(n_units(j-1)+1);
end
d_w = index_track(L);      % Total number of weights up to (and including) the last layer
d_beta = n_units(L)+1;     % Dimension of the weights beta connecting the last layer to the output
d_theta = d_w+d_beta;      % Total number of parameters

% Initialise weights and set initial mu equal to initial weights
layers = [size(X_train,2) n_units 1];  % Full structure of NN -> [input,hidden,output]
weights = nnInitialize(layers);
mu=[];
for i=1:length(layers)-1
    mu=[mu;weights{i}(:)];
end
% Initialize b and c
b = normrnd(0,cScale,d_theta,1);
if isotropic 
    c = cScale;
else
    c = cScale*ones(d_theta,1);
end
% Initialize lambda
lambda=[mu;b;c];

% Hyperparameters for inverse-Gamma prior on sigma2
alpha0_sigma2 = 0; 
beta0_sigma2 = 0; 

W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1; 
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end
beta = mu(d_w+1:d_theta);
alpha_sigma2 = alpha0_sigma2+length(y_train)/2; % Optimal VB parameter for updating sigma2 
beta_sigma2 = alpha_sigma2;                     % Mean_sigma2 and mean_sigma2_inverse are 
                                                % Initialised at small values 1/2 and 1 respectively  
mean_sigma2_inverse = alpha_sigma2/beta_sigma2;
mean_sigma2 = beta_sigma2/(alpha_sigma2-1);
mean_sigma2_save(1) = mean_sigma2;
% Compute prediction loss
[PPS_current,MSE_current] = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr,mean_sigma2);
disp(['Initial MSE: ',num2str(MSE_current)]);
MSE_DL(1) = MSE_current;
PPS_DL(1) = PPS_current;

shrinkage_gamma = .01*ones(p,1); % Initialise gamma_beta, the shrinkage parameters
mu_tau = zeros(p,1);             % Parameters for the auxiliary tau_j
mu_matrixW1_tilde = reshape(mu(W1_tilde_index),n_units(1),p);
b_matrixW1_tilde = reshape(b(W1_tilde_index),n_units(1),p);
if isotropic
    for j = 1:p
        mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
            b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+c^2*n_units(1);
        mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);        
    end
    lambda_tau = shrinkage_gamma.^2;
else
    c_matrixW1_tilde = reshape(c(W1_tilde_index),n_units(1),p);
    for j = 1:p
        mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
            b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+sum(c_matrixW1_tilde(:,j).^2);
        mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
    end
    lambda_tau = shrinkage_gamma.^2;
end
mean_inverse_tau = mu_tau;              % VB mean <1/tau_j>
shrinkage_gamma_seq = shrinkage_gamma;  %

minibatch = datasample(data,batchsize);
y = minibatch(:,1);
X = minibatch(:,2:end);

rqmc = normrnd_qmc(S,d_theta+1);        % Using quasi MC random numbers  
% grad_g_lik_store = zeros(S,)
for s=1:S
    U_normal = rqmc(s,:)';
    epsilon1=U_normal(1);
    epsilon2=U_normal(2:end);
    theta=mu+epsilon1*b+c.*epsilon2;   

    W_seq = cell(1,L);        
    W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1;
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
    end
    beta = theta(d_w+1:d_theta);    
    
    W1_tilde = W1(:,2:end);                 % Weights without biases
    W1_tilde_gamma = W1_tilde*diag(mean_inverse_tau);
    grad_prior_w_beta = [zeros(n_units(1),1);-W1_tilde_gamma(:);zeros(d_w+d_beta-index_track(1),1)];        
        
    grad_llh = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse);
     
    grad_h = grad_prior_w_beta+grad_llh;    % Gradient of log prior plus log-likelihood
    grad_log_q = vbGradientLogq(b,c,theta,mu,isotropic);
    grad_theta = grad_h-grad_log_q;
    
    grad_g_lik_store(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';
end
grad_lb = (mean(grad_g_lik_store))';
gradient_lambda = vbNaturalGradient(b,c,grad_lb,isotropic);
gradient_bar = gradient_lambda;

% Prepare parameters for training
idxEpoch = 0;          % Index of current epoch
iter = 1;              % Index of current iteration
stop = false;          % Stop flag for early stopping
lambda_best = lambda;  % Store optimal lambda for output
idxPatience = 0;       % Index of number of consequent non-decreasing iterations
                       % for early stopping
disp('---------- Training Phase ----------')
while ~stop 
    iter = iter+1;
    % Display training results after each 'verbose' iteration
    if (verbose && ~mod(iter,verbose))
        if(~mod(iter,num1Epoch))
            idxEpoch = idxEpoch + 1;
            disp(['Epoch: ',num2str(idxEpoch)]);
        end
        disp(['Current MSE: ',num2str(MSE_current)]);
    end
%     minibatch = datasample(data,batchsize);
    for k=1:5
        [X,y,X_val,y_val] = splitData(X,y,k,5);
        rqmc = normrnd_qmc(S,d_theta+1);
        for s=1:S
            U_normal = rqmc(s,:)';
            epsilon1=U_normal(1);
            epsilon2=U_normal(2:end);
            theta=mu+b*epsilon1+c.*epsilon2;   

            W_seq = cell(1,L);        
            W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
            W_seq{1} = W1;
            for j = 2:L
                index = index_track(j-1)+1:index_track(j);
                Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
                W_seq{j} = Wj; 
            end
            beta = theta(d_w+1:d_w+d_beta);    

            W1_tilde = W1(:,2:end); % Weights without biases
            W1_tilde_gamma = W1_tilde*diag(mean_inverse_tau);
            grad_prior_w_beta = [zeros(n_units(1),1);-W1_tilde_gamma(:);zeros(d_w+d_beta-index_track(1),1)];        

            grad_llh = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse);

            grad_h = grad_prior_w_beta+grad_llh;
            grad_log_q = vbGradientLogq(b,c,theta,mu,isotropic);
            grad_theta = grad_h-grad_log_q;

            grad_g_lik_store(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';
        end
        grad_lb = (mean(grad_g_lik_store))';
        gradient_lambda = vbNaturalGradient(b,c,grad_lb,isotropic);

        grad_norm = norm(gradient_lambda);
        norm_gradient_threshold = 100;
        if norm(gradient_lambda)>norm_gradient_threshold
            gradient_lambda = (norm_gradient_threshold/grad_norm)*gradient_lambda;
        end

        gradient_bar_old = gradient_bar;
        gradient_bar = grad_weight*gradient_bar+(1-grad_weight)*gradient_lambda;     

        if iter>tau
            stepsize=lrate*tau/iter;
        else
            stepsize=lrate;
        end

        lambda=lambda+stepsize*gradient_bar;

        mu=lambda(1:d_theta,1);
        b=lambda(d_theta+1:2*d_theta,1);
        c=lambda(2*d_theta+1:end);

        W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
        W_seq{1} = W1; 
        for j = 2:L
            index = index_track(j-1)+1:index_track(j);
            Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
            W_seq{j} = Wj; 
        end
        beta = mu(d_w+1:d_theta);

    % Update tau and shrinkage parameters    
    %     if mod(iter,10) == 0
         if mod(iter,1) == 0
            mu_matrixW1_tilde = reshape(mu(W1_tilde_index),n_units(1),p);
            b_matrixW1_tilde = reshape(b(W1_tilde_index),n_units(1),p);
            if isotropic
                for j = 1:p
                    mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
                        b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+c^2*n_units(1);
                    mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
                    lambda_tau(j) = shrinkage_gamma(j)^2;
                end
            else
                c_matrixW1_tilde = reshape(c(W1_tilde_index),n_units(1),p);
                for j = 1:p
                    mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
                        b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+sum(c_matrixW1_tilde(:,j).^2);
                    mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
                    lambda_tau(j) = shrinkage_gamma(j)^2;
                end
            end
            mean_inverse_tau = mu_tau;
            mean_tau = 1./mu_tau+1./lambda_tau;
            shrinkage_gamma = sqrt((n_units(1)+1)./mean_tau);
            shrinkage_gamma_seq = [shrinkage_gamma_seq,shrinkage_gamma];
        end

    % Update VB posterior for sigma2, which is inverse Gamma    
    %     if (iter >= 1000)&&(mod(iter,100) == 0)
        if (iter >= 1)&&(mod(iter,1) == 0)     
            beta_sigma2 = beta0_sigma2+sumResidualSquared(y_train,X_train,W_seq,beta)/2;
    %         beta_sigma2 = beta0_sigma2+sumResidualSquared(y,X,W_seq,beta)/2;
            mean_sigma2_inverse = alpha_sigma2/beta_sigma2;
            mean_sigma2 = beta_sigma2/(alpha_sigma2-1);
            mean_sigma2_save = [mean_sigma2_save,mean_sigma2];
        end
        [PPS_current,MSE_current] = deepGLMpredictLoss(X_val,y_val,W_seq,beta,distr,mean_sigma2);
    end
  
    MSE_DL(iter) = MSE_current;
    PPS_DL(iter) = PPS_current;

    if MSE_DL(iter)>=MSE_DL(iter-1)
        gradient_bar = gradient_bar_old;
    end
    
    if MSE_DL(iter)<=min(MSE_DL)
        lambda_best = lambda;
        idxPatience = 0;
        idxbest = iter;
    else
        idxPatience = idxPatience+1;
    end
    
    if (idxPatience>patience)||(idxEpoch>epoch) 
        stop = true; 
    end 
end
disp('---------- Training Completed! ----------')
disp(['Number of iteration:',num2str(iter)]);
disp(['PPS best: ',num2str(min(PPS_DL))]);
disp(['MSE best: ',num2str(min(MSE_DL))]);
% disp(['Sigma best: ',num2str(sqrt(mean_sigma2_save(idxbest)))]);

% Extract mode of weights to make prediction
lambda = lambda_best;
mu = lambda(1:d_theta,1);
b = lambda(d_theta+1:2*d_theta,1);
c = lambda(2*d_theta+1:end);
if isotropic              % For isotropic structure
    SIGMA = b*b' + c^2*eyes(d_theta);
else
    SIGMA = b*b' + diag(c.^2);
end
    
W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1; 
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end
beta = mu(d_w+1:d_w+d_beta);

% Store output in a struct
est.out.weights = W_seq; 
est.out.beta = beta;
est.out.shrinkage = shrinkage_gamma_seq;
est.out.iteration = iter;
est.out.mse = MSE_DL;
est.out.pps = PPS_DL;
est.out.vbMU = mu;            % Mean of variational distribution of weights
est.out.b = b;
est.out.c = c;
est.out.vbSIGMA = SIGMA;      % Covariance matrix of variational distribution 
                              % of weights
est.out.nparams = d_theta;    % Number of parameters     
est.out.indexTrack = index_track;
est.out.sigma2Alpha = alpha_sigma2;
est.out.sigma2Beta = beta_sigma2;
% est.out.idxBest = idxbest;
est.out.sigma2Mean = mean_sigma2_save(end);
est.out.sigma2MeanIter = mean_sigma2_save;
end