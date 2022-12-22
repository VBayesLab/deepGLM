# Function to calculate activation function
# Input must be a matrix
nnActivation <- function(a,func){
  switch (func,
          Linear = {out <- a},
          Sigmoid = {out <- 1/(1+exp(-a))},
          ReLU = {out <- pmax(a,0)},
          defaut)
  return(out)
}
