# Function to calculate lowerbound of variational distribution
vbLowerBound <- function(b,c,distr,p,beta_sigma2,alpha_sigma2,alpha0_sigma2,beta0_sigma2,
                         mean_sigma2_inverse,n_units,shrinkage_gamma,mean_tau,datasize,
                         lambda_tau,d_w_tilde,shrinkage_l2,mean_w_tilde,mean_column_j_tilde,
                         mean_inverse_tau){
  if(distr=="normal"){
    mean_log_sig2 <- log(beta_sigma2)-digamma(alpha_sigma2)
    logdet <- log(det(1 + (b/(c^2)) %*% b)) + sum(log(c^2))
    constMean <- -(alpha0_sigma2+1)*mean_log_sig2 - beta0_sigma2*mean_sigma2_inverse+
      0.5*sum(2*(n_units[1]+1)*log(shrinkage_gamma)-(shrinkage_gamma^2)*mean_tau)-
      0.5*datasize*mean_log_sig2+
      lgamma(alpha_sigma2)-alpha_sigma2*log(beta_sigma2)+
      (alpha_sigma2+1)*mean_log_sig2+alpha_sigma2-
      0.5*(sum(log(lambda_tau))-p) + 0.5*logdet +
      0.5*d_w_tilde*log(shrinkage_l2) - 0.5*shrinkage_l2*mean_w_tilde -
      0.5*sum(c(mean_column_j_tilde)*c(mean_inverse_tau))
  }else{
    logdet = log(det(1 + (b/(c^2)) %*% b)) + sum(log(c^2))
    constMean = 0.5*sum(2*(n_units[1]+1)*log(shrinkage_gamma)-
                          (shrinkage_gamma^2)*mean_tau)-0.5*(sum(log(lambda_tau))-p)+
      0.5*logdet+0.5*d_w_tilde*log(shrinkage_l2) -
      0.5*shrinkage_l2*mean_w_tilde-
      0.5*sum(c(mean_column_j_tilde)*c(mean_inverse_tau))
  }
  return(constMean)
}
