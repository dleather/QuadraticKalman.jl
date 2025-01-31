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

# DESCRIPTION : Quadratic Kalman smoothing with the QKF.
#-------------------------------------------------------
QKS <- function(QKF.object) {

      
      #=========================#
      #======= WARNING =========#
      #=========================#
      #                         #
      # QKF.object must be the  #
      # list resulting from the #####
      # filtering with QKF function.#
      #                             #
      #=============================#
      #=============================#
      #=============================#
      
      # 0. Loading needed package.
      #===========================
      require("matrixcalc")
      library(matrixcalc)
      
      # 1. Validity Checkings.
      #=======================
      
      # 1.1 Checking of argument
      if(class(QKF.object)!="list" |
               names(QKF.object)[1]!= "loglik.vector" |
               names(QKF.object)[2]!= "Z.updated" |
               names(QKF.object)[3]!= "P.updated" |
               names(QKF.object)[4]!= "Z.predicted" |
               names(QKF.object)[5]!= "P.predicted" |
               names(QKF.object)[6]!= "Y.predicted" |
               names(QKF.object)[7]!= "M.predicted" |
               names(QKF.object)[8]!= "Phi.tilde.t"
               
      ) {
            stop("argument MUST be the rough list resulting from the QKF function.")
      }
      
      
      # 1.2 Taking arguments
      n.Z <- nrow(QKF.object$Z.updated)
      n <- (-1+sqrt(1+4*n.Z))/2
      n.vech <- n*(n+3)/2
      time.length <- ncol(QKF.object$Z.updated)
      
      # 1.3 Building selection matrix
      selection.matrix <- matrix(0, nrow = n, ncol = n.Z)
      selection.matrix[,(1:n)] <- diag(1, n)
      
      if (n==1){
            selection.block <- matrix(1,1,1)
      } else{
            selection.block <- elimination.matrix(n)
      }
      selection.matrix <- rbind(selection.matrix,
                                cbind(
                                      matrix(0, nrow = nrow(selection.block), ncol = n),
                                      selection.block))
      
      # 1.3 Building duplication matrix
      dupli.matrix <- matrix(0, nrow = n.Z, ncol = n)
      dupli.matrix[(1:n),] <- diag(1, n)
      
      if (n==1){
            dupli.block <- matrix(1,1,1)
      } else{
            dupli.block <- duplication.matrix(n)
      }
      dupli.matrix <- cbind(dupli.matrix,
                                rbind(
                                      matrix(0, nrow = n, ncol = ncol(dupli.block)),
                                      dupli.block))
      
      # 2. Initializing. 
      #=================
      Z.smoothed <- matrix(0, nrow = n.vech, ncol = time.length)
      P.smoothed <- array(0, dim = c(n.vech, n.vech, time.length))
      smoothing.gain <- array(0, dim = c(n.vech, n.vech, time.length))
      
      Z <- selection.matrix %*% QKF.object$Z.updated[,time.length]
      P <- selection.matrix %*% QKF.object$P.updated[,,time.length] %*% t(selection.matrix)
      Z.smoothed[,time.length] <- Z
      P.smoothed[,,time.length] <- P
      
      # 3. Smoothing iterations.
      #=========================
      for (t in (time.length-1):1) {
            
            # 3.1 Iterations
            P.predicted <- selection.matrix %*% QKF.object$P.predicted[,,(t+1)] %*%t(selection.matrix)
            P.updated <- selection.matrix %*% QKF.object$P.updated[,,t] %*%t(selection.matrix)
            reducted.Phi.tilde <- selection.matrix %*% QKF.object$Phi.tilde[,,t] %*% dupli.matrix
            Z.predicted <- selection.matrix %*% QKF.object$Z.predicted[,(t+1)]
            Z.updated <- selection.matrix %*% QKF.object$Z.updated[,t]
            
            if (abs(det(P.predicted))>1e-10) {
                  smoothing.gain[,,t] <- P.updated %*% 
                        t(reducted.Phi.tilde) %*% 
                        solve(P.predicted)
            } else {stop(cat("Non invertible matrix P predicted at time", t+1))}
            
            Z.smoothed[,t] <- Z.updated +
                  smoothing.gain[,,t] %*% 
                  (Z - Z.predicted)
            
            P.smoothed[,,t] <- P.updated + 
                  smoothing.gain[,,t] %*%
                  (P - P.predicted) %*%
                  t(smoothing.gain[,,t])
            
            
            # 3.2 Updating Z and P
            Z <- Z.smoothed[,t]
            P <- P.smoothed[,,t]
            
      }
      
      # 4. return results.
      #===================
      result <- list(Z.smoothed = Z.smoothed,
                     P.smoothed = P.smoothed,
                     smoothing.gain = smoothing.gain)
      return(result)
      
}