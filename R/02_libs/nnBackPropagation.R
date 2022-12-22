# Function to calculate backbrop of a DFNN
# Input:
#         X,y -> Matrix
#         W_seq -> list of matrices
#         beta -> vector
#         distr -> character
nnBackPropagation <- function(X,y,W_seq,beta,distr){
  output = list()
  n_train <- nrow(X)             # Number of mini-batch training observation
  L <- length(W_seq)             # Number of hidden layers until the last layer
  a_seq <- vector("list", length = L)
  Z_seq <- vector("list",length = L)

  a_seq[[1]] <- W_seq[[1]] %*% t(X)
  Z_seq[[1]] <- rbind(matrix(1,1,n_train),nnActivation(a_seq[[1]],"ReLU"))

  for(j in 2:L){
    a_seq[[j]] <- W_seq[[j]] %*% Z_seq[[j-1]]
    Z_seq[[j]] <- rbind(matrix(1,1,n_train),nnActivation(a_seq[[j]],"ReLU"))
  }
  delta_seq = vector("list", length = L+1)

  # Calculate error at the output layers according to distribution family of response
  nnOut = beta %*% Z_seq[[L]]
  switch(distr,
         normal = {delta_seq[[L+1]] <- t(y) - nnOut},
         binomial = {p_i <- 1/(1+exp(-nnOut))
         delta_seq[[L+1]] <- t(y) - p_i},
         poisson = {delta_seq[[L+1]] <- t(y) - exp(nnOut)},
         default)
  delta_seq[[L]] <- (beta[2:length(beta)] %*% delta_seq[[L+1]]) * nnActivationGrad(a_seq[[L]],"ReLU")

  for (j in (L-1):1) {
    Wj_tilde <- W_seq[[j+1]]
    Wj_tilde <- Wj_tilde[,2:ncol(Wj_tilde)]
    delta_seq[[j]] <- nnActivationGrad(a_seq[[j]],"ReLU")*(t(Wj_tilde) %*% delta_seq[[j+1]])
  }
  gradient_W1 <- delta_seq[[1]] %*% X
  gradient <- c(gradient_W1)
  # dim(gradient) <- c(ncol(gradient)*nrow(gradient),1)
  for (j in 2:L) {
    gradient_Wj <- c(delta_seq[[j]] %*% t(Z_seq[[j-1]]))
    # dim(gradient_Wj) <- c(ncol(gradient_Wj)*nrow(gradient_Wj),1)
    gradient <- c(gradient,gradient_Wj)
  }
  gradient <- c(gradient,c(Z_seq[[L]] %*% t(delta_seq[[L+1]])))
  output$gradient <- gradient
  output$nnOut <- nnOut
  return(output)
}
