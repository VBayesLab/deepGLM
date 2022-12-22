# Function to calculate sum square error of 2 vector
nnSumResidualSquare <- function(y,X,W_seq,beta){
  nnet_output <- nnFeedForward(X,W_seq,beta)        # Output vector of NN
  S <- sum((y-nnet_output)^2)
  return(S)
}
