# Function to make prediction on an unseen data using a trained DeepGLM model
# Input:
deepGLMpredict <- function(mdl,X,y=NULL,Interval=0,Nsample=1000,Intercept=TRUE){

  # Transform X to a row matrix
  if(is.numeric(X)){
    X <- rbind(X)
  }

  # If y is specify, check y

  # Store Nsample to mdl
  mdl$Nsample <- Nsample

  # If training data does not include intercepts, the add intercepts
  N <- nrow(X)                     # Number of observation in test data
  if(Intercept){
    X <- cbind(matrix(1,N,1),X)
  }

  alpha <- Interval

  # Load deepGLM params from struct
  W_seq <- mdl$out.weights
  beta <- mdl$out.beta
  distr <- mdl$dist

  # Calculate Neuron Network output
  nnet_output <- nnFeedForward(X,W_seq,beta)              # Output vector of NN
  out <- list()

  if(distr=="normal"){
    out$yhat = nnet_output                                # Prediction for continuous response
    # If ytest if provided, then calculate pps and mse
    if(length(y)>0){
      sigma2 <- mdl$out.sigma2Mean
      mse <- mean((y-nnet_output)^2)
      pps <- 1/2*log(sigma2) + 1/2/sigma2*mse
      out$mse <- mse
      out$pps <- pps
    }
    # Calculate confidence interval if required
    if(alpha!=0){
      interval <- predictionInterval(mdl,X,alpha)
      out$interval <- interval$interval
      out$yhatMatrix <- interval$yhatMC
    }

  }else if(distr=="binomial"){
    out$yNN <- nnet_output
    out$yProb <- exp(nnet_output)/(1+exp(nnet_output))
    y_pred <- as.numeric(nnet_output>0)                        #  Prediction for binary response
    out$yhat <- y_pred
    #If ytest if provided, then calculate pps and mse
    if(length(y)>0){
      pps <- mean(-y*nnet_output+log(1+exp(nnet_output)))
      cr <- mean(y==y_pred)                                #  Miss-classification rate
      out$pps <- pps
      out$accuracy <- cr
    }

  }else if(distr=="poisson"){
    out$yNN <- nnet_output
    y_pred <- exp(nnet_output)                            #  Prediction for poisson response
    out$yhat <- y_pred
    if(length(y)>0)
      pps <- mean(-y*nnet_output+exp(nnet_output))
    mse <- mean((y-y_pred)^2)
    out$mse <- mse
    out$pps <- pps

  }else{
    message("Distribution must be: normal, binomial, poisson")
  }
  return(out)
}
