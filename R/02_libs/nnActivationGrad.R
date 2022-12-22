# Function to calculate derivative of activation function
# Input must be a matrix
nnActivationGrad <- function(a,func){
  switch (func,
          Linear = {out <- matrix(1,nrow(a),ncol(a))},
          Sigmoid = {out <- 1/(1+exp(-a))},
          ReLU = {out <- (a>0)*1},
          defaut)
  return(out)
}
