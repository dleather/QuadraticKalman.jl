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

# DESCRIPTION : builds the conditional variance covariance of the augmented model.
#---------------------------------------------------------------------------------
augmented.condi.vcov <- function(Z, Mu, Phi, Omega, Lambda) {
      
      n <- nrow(Mu)
      k <- ncol(Omega)
      
      X <- matrix(Z[1:n], nrow = n, ncol = 1)
      In <- diag(1,n)
      
      Gamma <- (In %x% Mu) + (Mu %x% In) + (In %x% (Phi %*% X)) + ((Phi %*% X) %x% In)
      Sigma <- tcrossprod(Omega)
      
      
      block1.1 <- Sigma
      block1.2 <- Sigma %*% t(Gamma) 
      block2.1 <- t(block1.2)
      
      block2.2.cst <- matrix((diag(1,k^2) + Lambda) %*% (Sigma %x% Sigma), ncol=1)
      block2.2.Z <- ((diag(1,k^2) + Lambda) %x% (diag(1,k^2) + Lambda)) %*%
            (In %x% Lambda %x% In) %*% (matrix(Sigma, ncol = 1) %x% diag(1, k^2)) %*% (Mu %x% Mu + 
                  (cbind(Mu %x% Phi + Phi %x% Mu, Phi %x% Phi) %*% Z))
      
      sum.block2.2 <- block2.2.cst + block2.2.Z
      block2.2 <- matrix(sum.block2.2, n^2, n^2)
      
      result <- rbind(cbind(block1.1, 
                            block1.2),
                      cbind(block2.1,
                            block2.2)
      )
      return(result)
}



# DESCRIPTION : function for building the augmented state-space model from the linear-quadratic form.
#----------------------------------------------------------------------------------------------------
augmented.transition <- function(Z, Mu, Phi, Omega, Lambda) {
      
      n <- length(Mu)
      Mu <- as.matrix(Mu)
      
      Mu.tilde <- rbind(Mu, 
                        matrix(Mu %*% t(Mu) + Omega %*% t(Omega) , ncol = 1)
      )
      
      Phi.tilde <- rbind(cbind(Phi, matrix(0, nrow = n, ncol = n^2)),
                         cbind(Mu %x% Phi + Phi %x% Mu, Phi %x% Phi)
      )
      
      Sigma.tilde <- augmented.condi.vcov(Z, Mu, Phi, Omega, Lambda)
      
      return(list(Mu.tilde = Mu.tilde, 
                  Phi.tilde = Phi.tilde, 
                  Sigma.tilde = Sigma.tilde))
}

augmented.measurement <- function(A, B, C, D) {
      
      m <- length(A)
      n <- ncol(B)
      
      Vec.C <- matrix(0, nrow = m, ncol = n^2)
      for (i in 1:m) {
            Vec.C[i,] <- c(C[,,i])
      }
      
      B.tilde <- cbind(B, Vec.C)
      
      return(list(A = A,
                  B.tilde = B.tilde,
                  D = D))
}