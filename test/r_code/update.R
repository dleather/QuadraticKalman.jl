######################################################################
###                                                                ###
### quadratic Kalman Filtering for quadratic measurement equations ###
### ============================================================== ###
###                                                                ###
### Authors:                                                       ###
### --------                                                       ###
### Alain Monfort, Jean-Paul Renne, and Guillaume Roussellet       ###
###                                                                ###
### Please quote the following article when using the filter:      ###
### ---------------------------------------------------------      ###
### "A Quadratic Kalman Filter" (2013)                             ###
###                                                                ###
######################################################################

# DESCRIPTION : This is the function for the updating step of the latent factors.
#--------------------------------------------------------------------------------
update.mean <- function(Z.pred, P.pred, Y.pred, M.pred, multi, observable) {
      Z.pred <- as.matrix(Z.pred)
      P.pred <- as.matrix(P.pred)
      Y.pred <- as.matrix(Y.pred)
      M.pred <- as.matrix(M.pred)
      
      # stability conditions
      if (abs(det(M.pred)) >1e-10 | sum(is.na(M.pred))==0 | sum(M.pred==Inf)==0) {
            
            Z.up <- Z.pred + P.pred %*% t(multi) %*% solve(M.pred) %*% (observable - Y.pred)
            
      } else {
            
            stop("variance of observable variables not invertible")
            
      }
      
      return(Z.up)
}

update.vcov <- function(P.pred, M.pred, multi) {
      P.pred <- as.matrix(P.pred)
      M.pred <- as.matrix(M.pred)
      
      # stability conditions
      if (abs(det(M.pred)) >1e-10 | sum(is.na(M.pred))==0 | sum(M.pred==Inf)==0) {
            
            P.up <- P.pred - P.pred %*% t(multi) %*% solve(M.pred) %*% multi %*% P.pred
            
      } else {
            
            stop("variance of observable variables not invertible")
            
      }
      
      return(P.up)
}
