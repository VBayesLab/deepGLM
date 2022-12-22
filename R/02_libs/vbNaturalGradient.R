# Function to calculate natural gradient
# Input:
#         b,c,grad -> vector
vbNaturalGradient <- function(b,c,grad){
  d <- length(b)
  grad1 <- grad[1:d]
  grad2 <- grad[(d+1):(2*d)]
  grad3 <- grad[(2*d+1):length(grad)]
  c2 <- c^2
  b2 <- b^2
  prod1 <- c(b %*% grad1)*b + (grad1*c2)
  const <- sum(b2/c2)
  const1 <- 0.5 + 0.5/const
  prod2 <- c(b %*% grad2)*b + (grad2*c2)
  prod2 <- const1*prod2
  alpha <- 1/(1+const)
  x <- alpha*b2/c^3
  y <- 1/c2 - 2*alpha*(b/c2)^2
  aux <- x/y
  prod3 <- 0.5*(grad3/y-c(1/((1+sum(x^2/y)))*(aux %*% grad3)) * aux)
  prod <- c(prod1,prod2,prod3)
  return(prod)
}
