# Calculate prediction interval for new observations
predictionInterval <- function(mdl,X,zalpha){
  predInterval <- list()
  # Load deepGLM params from struct
  Nsample <- mdl$Nsample
  mu <- mdl$out.vbMU
  SIGMA <- mdl$out.vbSIGMA
  n_units <- mdl$network
  index_track <- mdl$out.indexTrack
  alpha_sigma2 <- mdl$out.sigma2Alpha
  beta_sigma2 <- mdl$out.sigma2Beta

  # Calculate network parameters
  L <- length(n_units)                                   # Number of hidden layers
  p <- ncol(X)-1                                         # Number of covariates
  d_beta <- n_units[L]+1
  d_w <- index_track[L]

  yhat <- matrix(0,Nsample,nrow(X))                      # Predicted values of test data
  nnOut <- matrix(0,Nsample,nrow(X))                     # Output of NN
  W_seq <- vector("list",length = L)
  for (i in 1:Nsample) {
    theta_i <- rmvnorm(1,mean=mu,sigma=SIGMA)            # Generate samples of theta from Normal distribution
    sigma2_i <- 1/rgamma(1,alpha_sigma2,beta_sigma2)   # Generate samples of sigma from IG distribution

    # For each generated theta, restore neuron net structure
    W1 <- matrix(theta_i[1:index_track[1]],n_units[1],p+1)
    W_seq[[1]] <- W1
    for (j in 2:L){
      index <- (index_track[j-1]+1):index_track[j]
      Wj <- matrix(theta_i[index],n_units[j],n_units[j-1]+1)
      W_seq[[j]] <- Wj
    }
    beta <- theta_i[(d_w+1):(d_w+d_beta)]

    nnOut[i,] <- nnFeedForward(X,W_seq,beta)                      # Calculate neuron network output
    yhat[i,] <- rnorm(nrow(X),mean=nnOut[i,],sd=sqrt(sigma2_i))      # Calculate p(y|theta_i,sigma_i,X)
  }

  # 1-std prediction interval interval
  yhatLCL <- colMeans(yhat) - zalpha*apply(yhat, 2, sd)
  yhatUCL <- colMeans(yhat) + zalpha*apply(yhat, 2, sd)
  yhatInterval <- cbind(cbind(yhatLCL),cbind(yhatUCL))
  predInterval$yhatMC <- yhat
  predInterval$interval <- yhatInterval
  return(predInterval)
}
