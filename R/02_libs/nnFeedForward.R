# Function to calculate output of a DFNN
nnFeedForward <- function(X,W_seq,beta){
  n_train <- nrow(X)                                # Number of training observations
  # Make forward passes to all layers
  a <- W_seq[[1]] %*% t(X)
  Z <- rbind(matrix(1,1,n_train),nnActivation(a,"ReLU"))
  L <- length(W_seq)
  for (j in 2:L) {
    a <- W_seq[[j]] %*% Z
    Z <- rbind(matrix(1,1,n_train),nnActivation(a,"ReLU"))  # Add biases
  }
  nnOutput <- t(Z) %*% beta
  return(nnOutput)
}
