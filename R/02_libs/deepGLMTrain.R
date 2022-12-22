# Function to train deepGLM model
deepGLMTrain <- function(X_train,y_train,est){

  # Extract model parameters provided by users
  n_units <- est$network
  batchsize <- est$batchsize
  lrate <- est$lrate
  S <- est$S                      # Number of Monte Carlo samples to estimate the gradient
  tau <- est$tau                  # Threshold before reducing constant learning rate eps0
  grad_weight <- est$momentum     # Weight in the momentum
  cScale <- 0.01                 # Random scale factor to initialize b,c
  patience <- est$patience        # Stop if test error not improved after patience_parameter iterations
  epoch <- est$epoch              # Number of times learning algorithm scan entire training data
  verbose <- est$verbose
  distr <- est$dist
  LBwindow <- est$windowSize
  seed <- est$seed

  # Set random seed if specified
  if(!is.nan(seed)){
    set.seed(seed)
    # set.generator("MersenneTwister", initialization="init2002", resolution=53, seed=seed)
  }

  # Data merge for mini-batch sampling
  data <- cbind(y_train,X_train)
  datasize <- nrow(X_train)
  num1Epoch <- round(datasize/batchsize)    # Number of iterations per epoch

  # Network parameters
  L <- length(n_units)                      # Number of hidden layers
  p <- ncol(X_train)-1                      # Number of covariates
  W_seq <- vector("list",length = L)        # Cells to store weight matrices
  index_track <- numeric(L)                 # Keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
  index_track[1] <- n_units[1]*(p+1)        # Size of W1 is m1 x (p+1) with m1 number of units in the 1st hidden layer
  W1_tilde_index <- c((n_units[1]+1):index_track[1])  # Index of W1 without biases, as the first column if W1 are biases
  w_tilde_index <- c()                                # indices of non-biase weights, excluding W1, for l2-regulization prior
  for (j in 2:L) {
    index_track[j] <- index_track[j-1]+n_units[j]*(n_units[j-1]+1)
    w_tilde_index <- c(w_tilde_index,(index_track[j-1]+n_units[j]+1):index_track[j])
  }
  d_w <- index_track[L]                      # Total number of weights up to (and including) the last layer
  d_beta <- n_units[L]+1                     # Dimension of the weights beta connecting the last layer to the output
  d_theta <- d_w+d_beta                      # Total number of parameters
  w_tilde_index <- c(w_tilde_index,((d_w+2):d_theta))
  d_w_tilde <- length(w_tilde_index)

  # Initialise weights and set initial mu equal to initial weights
  layers <- c(ncol(X_train),n_units,1)       # Full structure of NN -> [input,hidden,output]
  weights <- nnInitialize(layers)
  mu <- c()                             # Mean of variational distribution
  for (i in 1:(length(layers)-1)) {
    temp <- weights[[i]]
    mu <- c(mu,c(temp))
  }

  # Initialize b and c and lambda
  b <- runif(d_theta, min=0, max=cScale)
  c <- cScale*rep(1,d_theta)
  lambda <- c(mu,b,c)

  # Separate weigths to 2 list: one for last hidden layers to output layer and for the rest
  W1 <- matrix(mu[1:index_track[1]],n_units[1],p+1)
  W_seq[[1]] <- W1
  for (j in 2:L) {
    index <- (index_track[j-1]+1):index_track[j]
    Wj <- matrix(mu[index],n_units[j],n_units[j-1]+1)
    W_seq[[j]] <- Wj
  }
  beta <- mu[(d_w+1):d_theta]

  # Get mini-batch
  idx <- sample.int(datasize,batchsize,replace = T)
  y <- y_train[idx,]
  X <- X_train[idx,]
  # X <- X_train
  # y <- y_train

  # Hyperparameters for inverse-Gamma prior on sigma2 if y~Nomal(0,sigma2)
  mean_sigma2_save <- c()
  if(distr == "normal"){
    alpha0_sigma2 <- 10
    beta0_sigma2 <- (alpha0_sigma2-1)*sd(y)
    alpha_sigma2 <- alpha0_sigma2 + 0.5*length(y_train)  # Optimal VB parameter for updating sigma2
    beta_sigma2 <- alpha_sigma2                          # Mean_sigma2 and mean_sigma2_inverse are
    # Initialised at small values 1/2 and 1 respectively
    mean_sigma2_inverse <- alpha_sigma2/beta_sigma2
    mean_sigma2 <- beta_sigma2/(alpha_sigma2-1)
    mean_sigma2_save[1] <- mean_sigma2
  }

  # Calculations for group Lasso coefficients
  shrinkage_gamma <- .01*rep(1,p)           # Initialise gamma_beta, the shrinkage parameters
  shrinkage_l2 <- .01                       # Hype-parameter for L2 prior
  mu_tau <- rep(0,p)                        # Parameters for the auxiliary tau_j
  mu_matrixW1_tilde <- matrix(mu[W1_tilde_index],n_units[1],p)
  b_matrixW1_tilde <- matrix(b[W1_tilde_index],n_units[1],p)

  c_matrixW1_tilde <- matrix(c[W1_tilde_index],n_units[1],p)
  for (j in 1:p) {
    mean_column_j_tilde <- mu_matrixW1_tilde[,j] %*% mu_matrixW1_tilde[,j] +
      b_matrixW1_tilde[,j] %*% b_matrixW1_tilde[,j] +
      sum(c_matrixW1_tilde[,j]^2)
    mu_tau[j] <- shrinkage_gamma[j]/sqrt(mean_column_j_tilde)
  }
  lambda_tau <- shrinkage_gamma^2
  mean_inverse_tau <- mu_tau                 # VB mean <1/tau_j>
  shrinkage_gamma_seq <- shrinkage_gamma
  mean_tau <- 1/mu_tau + 1/lambda_tau
  m <- n_units[1]

  # Prepare to calculate lowerbound
  if(distr=="normal"){
    const <- alpha0_sigma2*log(beta0_sigma2) - lgamma(alpha0_sigma2) -
      0.5*p*n_units[1]*log(2*pi) - 0.5*d_w_tilde*log(2*pi) -
      p*lgamma((n_units[1]+1)/2) - 0.5*datasize*log(2*pi) +
      p/2*log(2*pi) + 0.5*d_theta*log(2*pi) + d_theta/2
  }else{
    const <- -0.5*p*n_units[1]*log(2*pi) - 0.5*d_w_tilde*log(2*pi)-
      p*lgamma((n_units[1]+1)/2) + p/2*log(2*pi)+
      0.5*d_theta*log(2*pi) + d_theta/2
  }
  W1 <- matrix(mu[1:index_track[1]],n_units[1],p+1)
  W_seq[[1]] <- W1
  for (j in 2:L) {
    index <- (index_track[j-1]+1):index_track[j]
    Wj <- matrix(mu[index],n_units[j],n_units[j-1]+1)
    W_seq[[j]] <- Wj
  }
  beta <- mu[(d_w+1):d_theta]
  mu_w_tilde <- mu[w_tilde_index]
  b_w_tilde <- b[w_tilde_index]
  c_w_tilde <- c[w_tilde_index]
  mean_w_tilde <- c(mu_w_tilde %*% mu_w_tilde + b_w_tilde %*% b_w_tilde + sum(c_w_tilde^2))
  iter <- 1

  # calculate analytical terms of lowerbound
  constMean <- vbLowerBound(b,c,distr,p,beta_sigma2,alpha_sigma2,alpha0_sigma2,beta0_sigma2,
                            mean_sigma2_inverse,n_units,shrinkage_gamma,mean_tau,datasize,
                            lambda_tau,d_w_tilde,shrinkage_l2,mean_w_tilde,mean_column_j_tilde,
                            mean_inverse_tau)

  # Calculate gradient of lowerbound and lowerbound of the first iteration
  lb <- c()
  grad_g_lik_store <- matrix(0,S,3*d_theta)
  lb_iter <- matrix(0,1,S)
  iter <- 1
  gradient_lambda <- vbGradientLogLB(X,y,b,c,mu,S,p,L,d_theta,d_w,index_track,n_units,mean_inverse_tau,
                                     shrinkage_l2,datasize,distr,mean_sigma2_inverse,constMean,
                                     const,grad_g_lik_store,lb_iter,iter)
  gradient_bar <- gradient_lambda$gradient_lambda
  lb[iter] <- mean(gradient_lambda$lb_iter)/datasize
  cat("Initial LB: ", lb[iter],'\n')

  #--------------------------Training Phase-----------------------------
  # Prepare parameters for training
  idxEpoch <- 0                          # Index of current epoch
  iter <- 1                              # Index of current iteration
  stop <- FALSE                          # Stop flag for early stopping
  lambda_best <- lambda                  # Store optimal lambda for output
  idxPatience <- 0                       # Index of number of consequent non-decreasing
  # iterations for early stopping
  mean_column_j_tilde <- matrix(0,1,p)
  lb_bar <- c()

  print("---------- Training Phase ----------")
  while (!stop) {
    iter <- iter+1

    # Extract mini-batch
    idx <- sample.int(datasize,batchsize,replace = T)
    y <- y_train[idx,]
    X <- X_train[idx,]
    #    X <- X_train
    #    y <- y_train

    # Calculate analytical terms of lowerbound
    constMean <- vbLowerBound(b,c,distr,p,beta_sigma2,alpha_sigma2,alpha0_sigma2,beta0_sigma2,
                              mean_sigma2_inverse,n_units,shrinkage_gamma,mean_tau,datasize,
                              lambda_tau,d_w_tilde,shrinkage_l2,mean_w_tilde,mean_column_j_tilde,
                              mean_inverse_tau)

    # Calculate Natural Gradient
    grad_lb <- vbGradientLogLB(X,y,b,c,mu,S,p,L,d_theta,d_w,index_track,n_units,mean_inverse_tau,
                               shrinkage_l2,datasize,distr,mean_sigma2_inverse,constMean,
                               const,grad_g_lik_store,lb_iter,iter)
    gradient_lambda = grad_lb$gradient_lambda
    lb[iter] <- mean(grad_lb$lb_iter)/datasize

    # Prevent exploding Gradient
    grad_norm <- sqrt(sum(gradient_lambda^2))
    norm_gradient_threshold <- 100
    if(grad_norm > norm_gradient_threshold){
      gradient_lambda <- (norm_gradient_threshold/grad_norm)*gradient_lambda
    }

    # Momentum gradient
    gradient_bar_old <- gradient_bar
    gradient_bar <- grad_weight*gradient_bar+(1-grad_weight)*gradient_lambda

    # Adaptive learning rate
    if(iter>tau){
      stepsize <- lrate*tau/iter
    }else{
      stepsize <- lrate
    }

    # Gradient ascend
    lambda <- lambda + stepsize*gradient_bar

    # Restore model parameters from variational parameter lambda
    mu <- lambda[1:d_theta]
    b <- lambda[(d_theta+1):(2*d_theta)]
    c <- lambda[(2*d_theta+1):length(lambda)]
    W1 <- matrix(mu[1:index_track[1]],n_units[1],p+1)
    W_seq[[1]] <- W1
    for (j in 2:L){
      index <- (index_track[j-1]+1):index_track[j]
      Wj <- matrix(mu[index],n_units[j],n_units[j-1]+1)
      W_seq[[j]] <- Wj
    }
    beta <- mu[(d_w+1):d_theta]

    # Update tau and shrinkage parameters
    if(iter%%1 == 0){
      mu_matrixW1_tilde <- matrix(mu[W1_tilde_index],n_units[1],p)
      b_matrixW1_tilde <- matrix(b[W1_tilde_index],n_units[1],p)
      c_matrixW1_tilde <- matrix(c[W1_tilde_index],n_units[1],p)
      for (j in 1:p) {
        mean_column_j_tilde[j] <- mu_matrixW1_tilde[,j] %*% mu_matrixW1_tilde[,j] +
          b_matrixW1_tilde[,j] %*% b_matrixW1_tilde[,j] +
          sum(c_matrixW1_tilde[,j]^2)
        mu_tau[j] <- shrinkage_gamma[j]/sqrt(mean_column_j_tilde[j])
        lambda_tau[j] <- shrinkage_gamma[j]^2
      }
      mean_inverse_tau <- mu_tau
      mean_tau <- 1/mu_tau + 1/lambda_tau
      shrinkage_gamma <- sqrt((n_units[1]+1)/mean_tau)
      shrinkage_gamma_seq <- cbind(shrinkage_gamma_seq,shrinkage_gamma)

      mu_w_tilde <- mu[w_tilde_index]
      b_w_tilde <- b[w_tilde_index]
      c_w_tilde <- c[w_tilde_index]
      mean_w_tilde <- c(mu_w_tilde %*% mu_w_tilde + b_w_tilde %*% b_w_tilde + sum(c_w_tilde^2))
    }

    # Update VB posterior for sigma2, which is inverse Gamma
    if(distr=="normal"){
      if (iter%%1 == 0){
        sum_squared <- nnSumResidualSquare(y_train,X_train,W_seq,beta)
        beta_sigma2 <- beta0_sigma2 + sum_squared/2
        mean_sigma2_inverse <- alpha_sigma2/beta_sigma2
        mean_sigma2 <- beta_sigma2/(alpha_sigma2-1)
        mean_sigma2_save <- c(mean_sigma2_save,mean_sigma2)
      }
    }

    # Using lowerbound for validation
    if(iter>LBwindow){
      lb_bar[iter-LBwindow] <- mean(lb[(iter-LBwindow+1):iter])
      if(lb_bar[length(lb_bar)]>=max(lb_bar)){
        lambda_best <- lambda
        idxPatience <- 0
      }else{
        idxPatience <- idxPatience + 1
      }
    }

    # Early stopping
    if((idxPatience>patience)||(idxEpoch>epoch)){
      stop <- TRUE
    }

    # Display epoch index whenever an epoch is finished
    if(iter%%num1Epoch==0){
      idxEpoch <- idxEpoch + 1
    }

    # Display training results after each 'verbose' iteration
    if (verbose && iter%%verbose==0){
      if(iter>LBwindow){
        cat("Epoch: ", idxEpoch,  "    -  Current LB: ",lb_bar[iter-LBwindow],"\n")
        # message("Epoch: ", idxEpoch,  "  -  Current LB: ",lb_bar[iter-LBwindow])
      }
      else{
        cat("Epoch: ", idxEpoch,  "-  Current LB: ",lb[iter],"\n")
        # message("Epoch: ", idxEpoch,  "   -  Current LB: ",lb[iter])
      }
    }
  }

  # Display Training Results
  print('---------- Training Completed! ----------')
  cat("Number of iteration: ",iter,'\n')
  cat("LBBar best: ",max(lb_bar),'\n')
  # message("Number of iteration: ",iter)
  # message("LBBar best: ",max(lb_bar))

  # Store training output
  lambda <- lambda_best
  mu <- lambda[1:d_theta]
  b <- lambda[(d_theta+1):(2*d_theta)]
  c <- lambda[(2*d_theta+1):length(lambda)]
  SIGMA = cbind(b) %*% b + diag(c^2)

  W1 <- matrix(mu[1:index_track[1]],n_units[1],p+1)
  W_seq[[1]] <- W1
  for (j in 2:L){
    index <- (index_track[j-1]+1):index_track[j]
    Wj <- matrix(mu[index],n_units[j],n_units[j-1]+1)
    W_seq[[j]] <- Wj
  }
  beta <- mu[(d_w+1):d_theta]

  # Store output in a struct
  est$out.weights <- W_seq
  est$out.beta <- beta
  est$out.shrinkage <- shrinkage_gamma_seq
  colnames(est$out.shrinkage) <- NULL
  est$out.iteration <- iter
  est$out.vbMU <- mu                      # Mean of variational distribution of weights
  est$out.b <- b
  est$out.c <- c
  est$out.vbSIGMA <- SIGMA                # Covariance matrix of variational distribution of weights
  est$out.nparams <- d_theta              # Number of parameters
  est$out.indexTrack <- index_track
  est$out.muTau <- mu_tau

  if(distr=="normal"){
    est$out.sigma2Alpha <- alpha_sigma2
    est$out.sigma2Beta <- beta_sigma2
    est$out.sigma2Mean <- mean_sigma2_save[length(mean_sigma2_save)]
    est$out.sigma2MeanIter <- mean_sigma2_save
  }
  est$out.lbBar <- lb_bar[2:length(lb_bar)]
  est$out.lb <- lb

  return(est)
}
