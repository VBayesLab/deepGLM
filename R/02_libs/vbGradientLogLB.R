# Function to calculate the estimation of gradient of lowerbound
vbGradientLogLB <- function(X,y,b,c,mu,S,p,L,d_theta,d_w,index_track,n_units,mean_inverse_tau,
                            shrinkage_l2,datasize,distr,mean_sigma2_inverse,constMean,
                            const,grad_g_lik_store,lb_iter,iter){
  gradllh_out <- list()
  out <- list()
  batchsize <- nrow(X)
  #  set.generator("MersenneTwister", initialization="init2002", resolution=53, seed=iter)
  rqmc <- matrix(rnorm(S*(d_theta+1),0,1),S,d_theta+1)
  for (s in 1:S) {
    # Calculate theta
    U_normal <- rqmc[s,]
    epsilon1 <- U_normal[1]
    epsilon2 <- U_normal[2:length(U_normal)]
    theta <- mu + epsilon1*b + c*epsilon2

    W_seq <- vector("list", length = L)
    W1 <- matrix(theta[1:index_track[1]],n_units[1],p+1)
    W_seq[[1]] <- W1
    W1_tilde <- W1[,2:ncol(W1)]                                  # weights without biases
    W1_tilde_gamma <- W1_tilde %*% diag(c(mean_inverse_tau))
    grad_prior_w_beta <- c(rep(0,n_units[1]),-c(W1_tilde_gamma))
    for (j in 2:L) {
      index <- (index_track[j-1]+1):index_track[j]
      Wj <- matrix(theta[index],n_units[j],n_units[j-1]+1)
      W_seq[[j]] <- Wj
      Wj_tilde <- Wj[,2:ncol(Wj)]
      grad_prior_Wj <- c(rep(0,n_units[j]),-shrinkage_l2 %*% c(Wj_tilde))
      grad_prior_w_beta <- c(grad_prior_w_beta,grad_prior_Wj)
    }
    beta <- theta[(d_w+1):d_theta]
    beta_tilde <- beta[2:length(beta)]                    # vector beta without intercept
    grad_prior_beta <- c(0,c(-shrinkage_l2 %*% beta_tilde))
    grad_prior_w_beta <- c(grad_prior_w_beta,grad_prior_beta)

    if (distr=="normal"){
      gradllh_out <- nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr,mean_sigma2_inverse)
    }else if(distr=="binomial"){
      gradllh_out <- nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr)
    }else if(distr=="poisson"){
      gradllh_out <- nnGradLogLikelihood(W_seq,beta,X,y,datasize,distr)
    }else{
      message("Distribution must be: normal, binomial, poisson")
    }
    grad_llh <- gradllh_out$gradient
    yNN <- gradllh_out$nnOut

    grad_h <- grad_prior_w_beta + grad_llh              # Gradient of log prior plus log-likelihood
    grad_log_q <- vbGradientLogq(b,c,theta,mu)
    grad_theta <- grad_h - grad_log_q
    grad_g_lik_store[s,] <- c(grad_theta,epsilon1*grad_theta, epsilon2*grad_theta)

    # Calculate Lowerbound
    if(distr=="normal"){
      lb_iter[s] <- constMean-0.5*mean_sigma2_inverse*sum((y-yNN)^2)*datasize/batchsize + const
    }else if(distr=="binomial"){
      lb_iter[s] <- constMean + sum(y*yNN - log(1+exp(yNN)))*datasize/batchsize + const
    }else if(distr=="poisson"){
      lb_iter[s] <- constMean + sum(y*yNN - exp(yNN))*datasize/batchsize + const
    }else{
      message("Distribution must be: normal, binomial, poisson")
    }
  }
  grad_lb <- colMeans(grad_g_lik_store)
  gradient_lambda <- vbNaturalGradient(b,c,grad_lb)
  out$lb_iter <- lb_iter
  out$gradient_lambda <- gradient_lambda

  return(out)
}
