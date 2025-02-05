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

# DESCRIPTION : calculating unconditional moments of augmented state variable.
#-----------------------------------------------------------------------------
initialize.filter <- function(Mu, Phi, Omega) {
      
      #=======================================================#
      #====================== WARNING ========================#
      #=======================================================#
      #                                                       #
      # This function calculates the unconditional moments    #
      # of the augmented stationary state process.            #
      # It shall not be used when parameters are time-varying #
      #                                                       #
      #=======================================================#
      #=======================================================#
      #=======================================================#
      
      # 1. Validity checkings
      #======================
      if(length(Phi)==1){Phi <- matrix(Phi,1,1)}
      
      
      if (!inherits(Mu, "matrix")){
            stop("Mu must be a (n * 1) matrix")
      }
      if (!inherits(Phi, "matrix")){
            stop("Phi must be a (n * n) matrix")
      }
      if (!inherits(Omega, "matrix")){
            stop("Omega must be a (n * k) matrix ")
      }
      
      n <- nrow(Phi)
      k <- ncol(Omega)
      
      if(nrow(Mu)!=n | ncol(Mu)!=1){
            stop("Mu must be a (n * 1) matrix (No time varying parameters allowed)")
      }
      
      Phi <- as.matrix(Phi)
      if (nrow(Phi)!=n | ncol(Phi)!=n) {
            stop("Phi must be a (n * n) matrix (No time varying parameters allowed)")
      }
      
      Omega <- as.matrix(Omega)
      if (nrow(Omega)!=n) {
            stop("Omega must be a (n * k) matrix (No time varying parameters allowed)")
      }
      
      # Check missing values
      if(sum(is.na(Mu))!=0 | sum(is.null(Mu))!=0 | sum(Mu==Inf)!=0 |
               sum(is.na(Phi))!=0 | sum(is.null(Phi))!=0 | sum(Phi==Inf)!=0 |
               sum(is.na(Omega))!=0 | sum(is.null(Omega))!=0 | sum(Omega==Inf)!=0 ) {
            stop("Check values of transition parameters: no Inf, Null, or NA allowed")
      }
      
      # Check stationarity of Phi
      Phi.eig <- abs(eigen(Phi)$values)
      if(sum(Phi.eig>=1)!=0) {
            warning("Phi is not stationary")
      }
      
      
      # 2. Constructing the augmented parameters.
      #==========================================
      Sigma <- tcrossprod(Omega)
      
      Z <- matrix(0, nrow = (n*(n+1)), ncol = 1)
      Lambda <- matrix(0, nrow = k^2, ncol = k^2)
      for (i in 1:k){
            e.i <- matrix(0, nrow = k, ncol = 1)
            e.i[i] <- 1
            for (j in 1:k){
                  e.j <- matrix(0, nrow = k, ncol = 1)
                  e.j[j] <- 1
                  Lambda <- Lambda + (e.i %*% t(e.j)) %x% (e.j %*% t(e.i))
            }
      }
      
      param.aug <- augmented.transition(Z, Mu, Phi, Omega, Lambda)
      Mu.tilde <- as.matrix(param.aug$Mu.tilde)
      Phi.tilde <- as.matrix(param.aug$Phi.tilde)
      
      # 3. Calculation of unconditional mean.
      #======================================
      uncondi.mean <- solve(diag(1, (n*(n+1))) - Phi.tilde) %*% Mu.tilde
      
      # 4. Construction of unconditional variance.
      #===========================================
      In <- diag(1,n)
      
      uncondi.Gamma <- (In %x% (Mu + Phi %*% uncondi.mean[1:n,1])) + 
            ((Mu + Phi %*% uncondi.mean[1:n,1]) %x% In)
      
      # Building the vec of block 2.2
      In2 <- diag(1, n^2)
      Vec.block2.2 <- (In2 %x% (In2 + Lambda)) %*% matrix((Sigma %x% Sigma), ncol = 1) +
            ((In2 + Lambda) %x% (In2 + Lambda)) %*% (In %x% Lambda %x% In) %*% 
            (matrix(Sigma, ncol = 1) %x% In2) %*% (Mu %x% Mu + Phi.tilde[((n + 1):(n^2 + n)),] %*% uncondi.mean)
      
      # restacking the blocks in a matrix
      block1.1 <- Sigma
      block1.2 <- Sigma %*% t(uncondi.Gamma)
      block2.1 <- t(block1.2)
      block2.2 <- matrix(Vec.block2.2, nrow = n^2, ncol = n^2)
      final.block <- rbind(cbind(
            block1.1, block1.2),
                           cbind(
            block2.1, block2.2)
      )
      
      # Computing the unconditional variance
      uncondi.vcov <- matrix(
            (solve( diag(1, ((n*(n+1))^2) ) - Phi.tilde %x% Phi.tilde) %*% 
                  matrix(final.block, ncol = 1)),
            nrow = (n*(n+1)), ncol = (n*(n+1))
      )
      
      
      
      # 5. Return results.
      #===================
      result <- list(Z0 = uncondi.mean,
                     V0 = uncondi.vcov)
      return(result)
}
