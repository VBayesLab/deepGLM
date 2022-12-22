# Function vbGradientLogq
# b,c,theta,mu -> vector
vbGradientLogq <- function(b,c,theta,mu){
  x <- theta-mu
  d <- b/c^2
  grad_log_q <- -x/c^2 + c((d%*%x)/(1+(d%*%b)))*d
}
