# Function to initialize weights for deepGLM
# Input: layers is a data array specifying (input+hidden) layers
# Ex: c(20,10,10)
nnInitialize <- function(layers){
  # stopifnot(is.integer(layers))                    # layer must be array of interger
  num_layer <- length(layers)-1
  w <- vector("list",length = num_layer)       # Initialize a list to store matrices of weights
  for (i in 1:num_layer) {
    b <- sqrt(6)/(layers[i]+layers[i+1])
    if(i==1){
      w[[i]] <- matrix(runif(layers[i+1]*(layers[i]),-b,b),layers[i+1],layers[i])    # Input layer already has bias
    }
    else{
      w[[i]] <- matrix(runif(layers[i+1]*(layers[i]+1),-b,b),layers[i+1],layers[i]+1)
    }
  }
  return(w)
}
