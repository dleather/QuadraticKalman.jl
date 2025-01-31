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

# DESCRIPTION : Filtering function.
#----------------------------------
QKF <- function(Z0, V0, Mu.t, Phi.t, Omega.t, A.t, B.t, C.t, D.t, observable) {
      
      # 0. Calling functions.
      #======================
      #setwd(directory)
      source('augmented_model.R')
      source("checking_arguments.R")
      source("loglik.R")
      source("predict.R")
      source("update.R")
      source("quadratic_kalman_smoothing.R")
      
      
      # 1. Validity checkings.
      #=======================

      # 1.1 Checking of arguments' classes.
      #------------------------------------
      transition.param.checked <- check.transition(Mu.t, Phi.t, Omega.t, observable)
      n <- transition.param.checked$n
      k <- ncol(Omega.t)
      
      check.initial.filter(Z0, V0, n)
      measurement.param.checked <- check.measurement(A.t, B.t, C.t, D.t, observable, n)
      
      # Transition parameters
      time.length <- transition.param.checked$time.length
      Mu.t <- transition.param.checked$Mu.t
      Phi.t <- transition.param.checked$Phi.t
      Omega.t <- transition.param.checked$Omega.t
      
      # Measurement parameters
      m <- measurement.param.checked$m
      A.t <- measurement.param.checked$A.t
      B.t <- measurement.param.checked$B.t
      C.t <- measurement.param.checked$C.t
      D.t <- measurement.param.checked$D.t
      
      
      # 1.2 Checking of missing values
      if(sum(is.na(Mu.t))!=0 |
               sum(is.na(Phi.t))!=0 |
               sum(is.na(Omega.t))!=0 |
               sum(is.na(A.t))!=0 |
               sum(is.na(B.t))!=0 |
               sum(is.na(C.t))!=0 |
               sum(is.na(D.t))!=0 |
               sum(is.na(observable))!=0) {
            stop("Missing values are not taken into account by the filter (yet).")
      }
      
      # 2. Initializing values.
      #========================
      
      # 2.1 Initializing final values
      Z.pred <- matrix(0, nrow = (n*(n+1)) , ncol = time.length)
      Y.pred <- matrix(0, nrow = m , ncol = time.length)
      P.pred <- array(0, dim = c((n*(n+1)),(n*(n+1)),time.length))
      M.pred <- array(0, dim = c(m,m,time.length))
      Z.up <- matrix(0, nrow = (n*(n+1)) , ncol = time.length)
      P.up <- array(0, dim = c((n*(n+1)),(n*(n+1)),time.length))
      
      loglik.vector <- rep(0, time.length)
      
      Phi.tilde.t <- array(0, dim = c((n*(n+1)),(n*(n+1)),time.length))
      
      # 2.2 Initializing filter values
      Z <- Z0
      P <- V0
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
      
      
      # 3. Iterations for the filter.
      #==============================
      for (t in 1:time.length) {
            
            # 3.1 construct the augmented model
            #----------------------------------
            Mu <- as.matrix(Mu.t[,,t])
            Phi <- as.matrix(Phi.t[,,t])
            Omega <- as.matrix(Omega.t[,,t])
            param.aug.transition <- augmented.transition(Z, Mu, Phi, Omega, Lambda)
            Phi.tilde.t[,,t] <- param.aug.transition$Phi.tilde
            
            A <- as.matrix(A.t[,,t])
            B <- as.matrix(B.t[,,t])
            C <- array(C.t[,,,t], dim = c(n,n,m))
            D <- as.matrix(D.t[,,t])
            param.aug.measurement <- augmented.measurement(A, B, C, D)
            
            
            # 3.2 perform prediction step
            #----------------------------
            Z.pred[,t] <- predict.mean(param.aug.transition$Mu.tilde,
                                       param.aug.transition$Phi.tilde,
                                       Z
            )
            P.pred[,,t] <- predict.vcov(param.aug.transition$Phi.tilde,
                                        P,
                                        param.aug.transition$Sigma.tilde
            )
            
            Y.pred[,t] <- predict.mean(param.aug.measurement$A,
                                       param.aug.measurement$B.tilde,
                                       Z.pred[,t]
            )
            M.pred[,,t] <- predict.vcov(param.aug.measurement$B.tilde,
                                        P.pred[,,t],
                                        tcrossprod(param.aug.measurement$D)
            )
            
            # 3.3 perform updating step
            #--------------------------
            Z.up[,t] <- update.mean(Z.pred[,t],
                                    P.pred[,,t],
                                    Y.pred[,t],
                                    M.pred[,,t],
                                    param.aug.measurement$B.tilde,
                                    observable[,t]
            )
            
            P.up[,,t] <- update.vcov(P.pred[,,t],
                                     M.pred[,,t],
                                     param.aug.measurement$B.tilde
            )
            
            # 3.4 Correcting update
            #----------------------
            implied.XX <- matrix(Z.up[((n+1):(n^2+n)),t], nrow = n, ncol = n)
            implied.vcov <- implied.XX - tcrossprod(Z.up[(1:n),t])
            eig <- eigen(implied.vcov)
            
            eig$values[eig$values<0] <- 0 # putting negative eigenvalues to 0
            implied.corrected.vcov <- eig$vectors %*% diag(c(eig$values)) %*% solve(eig$vectors)
            # resetting Z-values to be coherent
            Z.up[((n+1):(n^2+n)),t] <- c(implied.corrected.vcov + tcrossprod(Z.up[(1:n),t]))
            
            # 3.5 Calculating log-likelihood
            #-------------------------------
            loglik.vector[t] <- calculate.loglik(Y.pred[,t],
                                                 M.pred[,,t],
                                                 observable[,t]
            )
            
            # 3.6 Looping
            #------------
            Z <- Z.up[,t]
            P <- P.up[,,t]
            
            
      }
      
      # 4. Sending results back.
      #=========================
      
      result <- list(loglik.vector = loglik.vector,
                     Z.updated = Z.up,
                     P.updated = P.up,
                     Z.predicted = Z.pred,
                     P.predicted = P.pred,
                     Y.predicted = Y.pred,
                     M.predicted = M.pred,
                     Phi.tilde.t = Phi.tilde.t                     
      )
      return(result)
      
      
      
}