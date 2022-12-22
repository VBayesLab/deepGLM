# Function to calculate gradient of log-likelihood
nnGradLogLikelihood <- function(w_seq,beta,X,y,datasize,distr,mean_sigma2_inverse){
  output <- list()
  n = nrow(X)
  out <- nnBackPropagation(X,y,w_seq,beta,distr)
  back_prop <- out$gradient
  nnOut <- t(out$nnOut)

  switch (distr,
          normal = {gradient_theta <- mean_sigma2_inverse*back_prop
          gradient <- datasize/n*gradient_theta  },        # To compensate the variational lowerbound
          binomial = {gradient <- datasize/n*back_prop},
          poisson = {gradient <- datasize/n*back_prop},
          default)
  output$gradient <- gradient
  output$nnOut <- nnOut
  return(output)
}
