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


# DESCRIPTION : function for likelihood calculation
# -------------------------------------------------
calculate.loglik <- function(Y.pred, M.pred, observable) {
      Y.pred <- as.matrix(Y.pred)
      M.pred <- as.matrix(M.pred)
      
      # stability conditions
      if (det(M.pred) >1e-10 | sum(is.na(M.pred))==0 | sum(M.pred==Inf)==0) {
            
            residual <- (observable - Y.pred)
            loglik <- -.5*(length(Y.pred)*log(2*pi) + 
                                 log(det(M.pred)) + 
                                 t(residual) %*% solve(M.pred) %*% residual)
            
      } else {
            
            stop("variance of observable variables not invertible or not positive")
            
      }
      
      return(loglik)
}