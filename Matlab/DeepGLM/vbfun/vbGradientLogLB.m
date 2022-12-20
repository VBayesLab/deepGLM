%% Script to calculate natural gradient of lowerbound
%rqmc = normrnd_qmc(S,d_theta+1);     % Using quasi MC random numbers 
rqmc = normrnd(0,1,S,d_theta+1);
% rng(iter)
% rqmc = rand(S,d_theta+1);
for s=1:S
    U_normal = rqmc(s,:)';
    epsilon1=U_normal(1);
    epsilon2=U_normal(2:end);
    theta=mu+epsilon1*b+c.*epsilon2;   

    W_seq = cell(1,L);        
    W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1;
    W1_tilde = W1(:,2:end); % weights without biases
    W1_tilde_gamma = W1_tilde*diag(mean_inverse_tau);
    grad_prior_w_beta = [zeros(n_units(1),1);-W1_tilde_gamma(:)]; 
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
        Wj_tilde = Wj(:,2:end);
        grad_prior_Wj = [zeros(n_units(j),1);-shrinkage_l2*Wj_tilde(:)];        
        grad_prior_w_beta = [grad_prior_w_beta;grad_prior_Wj];
    end
    beta = theta(d_w+1:d_theta);    
    beta_tilde = beta(2:end); % vector beta without intercept
    grad_prior_beta = [0;-shrinkage_l2*beta_tilde];
    grad_prior_w_beta = [grad_prior_w_beta;grad_prior_beta];
    
    if(strcmp(distr,'normal'))    
        [grad_llh,yNN] = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse);
    else
        [grad_llh,yNN] = nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr);
    end 
    
    grad_h = grad_prior_w_beta+grad_llh;    % Gradient of log prior plus log-likelihood
    grad_log_q = vbGradientLogq(b,c,theta,mu,isotropic);
    grad_theta = grad_h-grad_log_q;
    grad_g_lik_store(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';
    
%   ------------------ lower bound ---------------------------------------
    if(lbFlag)
        if(strcmp(distr,'normal'))
            lb_iter(s) = constMean...
                        -0.5*mean_sigma2_inverse*sum((y-yNN).^2)*datasize/batchsize...
                        +const;
        elseif(strcmp(distr,'binomial'))
            lb_iter(s) = constMean...
                        +sum(y.*yNN - log(1+exp(yNN)))*datasize/batchsize...
                        +const;
        else
            lb_iter(s) = constMean...
                        +sum(y.*yNN - exp(yNN))*datasize/batchsize...
                        +const;
        end
    end
%   ----------------------------------------------------------------------
end
grad_lb = (mean(grad_g_lik_store))';
gradient_lambda = vbNaturalGradient(b,c,grad_lb,isotropic);