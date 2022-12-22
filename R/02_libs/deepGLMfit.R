deepGLMfit <- function(X,y, Lrate=0.01, Network=c(10,10) , BatchSize=5000,
                       S=10, LRateFactor=10000, Momentum=0.6, Patience=100,
                       MaxEpoch = 100, Verbose=10, Distribution="normal",
                       WindowSize=100, Seed=NaN, Intercept=TRUE){

  # Store training settings in a list
  est <- list()
  est$S <- S
  est$lrate <- Lrate
  est$epoch <- MaxEpoch
  est$tau <- LRateFactor
  est$patience <- Patience
  est$network <- Network
  est$dist <- Distribution
  est$seed <- Seed
  est$icept <- Intercept
  est$momentum <- Momentum
  est$verbose <- Verbose
  est$windowSize <- WindowSize

  # Check if inputs are corrected
  checkInput(X,y,est)

  # Calculate batch size
  if(BatchSize<=1){                  # If specified batchsize is a propotion
    BatchSize <- BatchSize * nrow(X)
  }
  if(BatchSize>=nrow(X)){
    BatchSize <- nrow(X)
  }
  est$batchsize <- BatchSize

  # Insert intercepts if Intercept=TRUE
  if(Intercept){
    X <- cbind(matrix(1,nrow(X),1),X)
  }

  #Start to train deepGLM
  y <- as.matrix(y)
  t_start <- Sys.time()
  est <- deepGLMTrain(X,y,est)
  t_stop <- Sys.time()
  est$out.CPU <- t_stop - t_start
  cat("Training time: ",est$out.CPU,'\n')

  return(est)
}
