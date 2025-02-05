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

# DESCRIPTION : Checking arguments.
#----------------------------------
check.initial.filter <- function(Z0, V0, n){
      # You can comment out the following checks if you're sure inputs are valid:
      # if(!inherits(Z0, "matrix")){stop("Z0 must be a (n*(n+1) * 1 ) matrix")}
      # if(!inherits(V0, "matrix")){stop("V0 must be a (n*(n+1) * n*(n+1) ) matrix")}
      # if(n!=floor(n)){
      #       stop("n must be an integer")
      # }
      
      # if(length(Z0)!= (n*(n+1))) {stop("size of Z0 must be n*(n+1)")}
      # if(nrow(V0)!=(n*(n+1)) | ncol(V0)!=(n*(n+1))) {stop("size of V0 must be n*(n+1)")}
      
      # if(sum(is.na(Z0))!=0 | 
      #          sum(is.na(V0))!=0 |
      #          sum(Z0==Inf)!=0 |
      #          sum(V0==Inf)!=0 |
      #          sum(is.null(Z0))!=0 |
      #          sum(is.null(V0))!=0 ) {
      #       stop("Check values of Z0 and V0: no Inf, Null, or NA allowed")
      # }
      
      # But keep the reshaping:
      Z0 <- matrix(Z0, nrow = (n*(n+1)), ncol = 1)
      V0 <- matrix(V0, nrow = (n*(n+1)), ncol = (n*(n+1)))
         
      return(list(Z0 = Z0, V0 = V0))
}


# DESCRIPTION : Checking arguments of transtion equation
#-------------------------------------------------------
check.transition <- function(Mu.t, Phi.t, Omega.t, observable) {
      
      if (length(Mu.t) == 1) { Mu.t <- matrix(Mu.t, 1, 1) }
      if (length(Phi.t) == 1) { Phi.t <- matrix(Phi.t, 1, 1) }
      if (length(Omega.t) == 1) { Omega.t <- matrix(Omega.t, 1, 1) }
      
      # Commented out type checks:
      # if (!inherits(Mu.t, "matrix") && !inherits(Mu.t, "array")) {
      #       stop("Mu.t must be either a (n * 1) matrix or a (n * 1 * T) array")
      # }
      # if (!inherits(Phi.t, "matrix") && !inherits(Phi.t, "array")) {
      #       stop("Phi.t must be either a (n * n) matrix or a (n * n * T) array")
      # }
      # if (!inherits(Omega.t, "matrix") && !inherits(Omega.t, "array")) {
      #       stop("Omega.t must be either a (n * k) matrix or a (n * k * T) array")
      # }
      # if (!inherits(observable, "matrix")) {
      #       stop("observable must be a (m * T) matrix")
      # }
      
      n <- ncol(Phi.t)
      time.length <- ncol(observable)
      
      # Commented out dimension validations:
      # if (nrow(Phi.t) != n) { stop("dimension of Phi.t must be (n * n * T) or (n * n)") }
      # if (nrow(Mu.t) != n) { stop("dimension of Mu.t must be (n * 1 * T) or (n * 1)") }
      # if (ncol(Mu.t) != 1) { stop("dimension of Mu.t must be (n * 1 * T) or (n * 1)") }
      # if (nrow(Omega.t) != n) { stop("dimension of Omega.t must be (n * k * T) or (n * k)") }
      
      # Check time-varying and perform reshaping as needed:
      if (is.na(dim(Mu.t)[3]) == TRUE || is.null(dim(Mu.t)[3]) == TRUE || dim(Mu.t)[3] == 1) {
            Mu.t <- array(1, c(1, 1, time.length)) %x% matrix(Mu.t, nrow = n, ncol = 1)
      } 
      if (is.na(dim(Phi.t)[3]) == TRUE || is.null(dim(Phi.t)[3]) == TRUE || dim(Phi.t)[3] == 1) {
            Phi.t <- array(1, c(1, 1, time.length)) %x% Phi.t
      } 
      if (is.na(dim(Omega.t)[3]) == TRUE || is.null(dim(Omega.t)[3]) == TRUE || dim(Omega.t)[3] == 1) {
            Omega.t <- array(1, c(1, 1, time.length)) %x% Omega.t
      }     
     
      # Commented out final time-varying dimension check:
      # if (dim(Mu.t)[3] != time.length || dim(Phi.t)[3] != time.length || dim(Omega.t)[3] != time.length) {
      #       stop("if transition parameters are time-varying, third dimension must be equal to the time length.")
      # }
      
      # Commented out missing/null/Inf checks:
      # if (sum(is.na(Mu.t)) != 0 || sum(is.null(Mu.t)) != 0 || sum(Mu.t == Inf) != 0 ||
      #     sum(is.na(Phi.t)) != 0 || sum(is.null(Phi.t)) != 0 || sum(Phi.t == Inf) != 0 ||
      #     sum(is.na(Omega.t)) != 0 || sum(is.null(Omega.t)) != 0 || sum(Omega.t == Inf) != 0) {
      #       stop("Check values of transition parameters: no Inf, Null, or NA allowed")
      # }
      
      result <- list(Mu.t = Mu.t, 
                     Phi.t = Phi.t,
                     Omega.t = Omega.t, 
                     n = n, 
                     time.length = time.length)
}

# DESCRIPTION : Checking arguments of measurement equation.
#----------------------------------------------------------
check.measurement <- function(A.t, B.t, C.t, D.t, observable, n) {
      
      m <- nrow(observable)
      time.length <- ncol(observable)
      
      if (length(A.t) == 1) { A.t <- matrix(A.t, 1, 1) }
      if (length(B.t) == 1) { B.t <- matrix(B.t, 1, 1) }
      if (length(C.t) == 1) { C.t <- array(C.t, dim = c(1, 1, 1)) }
      if (m == 1 & n > 1) { C.t <- array(C.t, dim = c(n, n, 1)) }
      if (length(D.t) == 1) { D.t <- matrix(D.t, 1, 1) }
      
      # Commented out type checks:
      # if (!inherits(A.t, "matrix") && !inherits(A.t, "array")){
      #       stop("A.t must be either a (m * 1) matrix or a (m * 1 * T) array")
      # }
      # if (!inherits(B.t, "matrix") && !inherits(B.t, "array")){
      #       stop("B.t must be either a (m * n) matrix or a (m * n * T) array")
      # }
      # if (!inherits(C.t, "array")){
      #       stop("C.t must be either a (n * n * m) array or a (n * n * m * T) array")
      # }
      # if (!inherits(D.t, "matrix") && !inherits(D.t, "array")){
      #       stop("D.t must be either a (m * k) matrix or a (m * k * T) array")
      # }
      # if (!inherits(observable, "matrix")){
      #       stop("observable must be a (m * T) matrix")
      # }
      # if (n != floor(n)){
      #       stop("n must be an integer")
      # }
      
      # Commented out dimension validations:
      # if (nrow(A.t) != m) { stop("dimension of A.t must be (m * 1 * T) or (m * 1)") }
      # if (ncol(A.t) != 1) { stop("dimension of A.t must be (m * 1 * T) or (m * 1)") }
      # if (nrow(B.t) != m) { stop("dimension of B.t must be (m * n * T) or (m * n)") }
      # if (ncol(B.t) != n) { stop("dimension of B.t must be (m * n * T) or (m * n)") }
      # if (nrow(C.t) != n) { stop("dimension of C.t must be (n * n * m * T) or (n * n * m)") }
      # if (ncol(C.t) != n) { stop("dimension of C.t must be (n * n * m * T) or (n * n * m)") }
      # if (dim(C.t)[3] != m) { stop("dimension of C.t must be (n * n * m * T) or (n * n * m)") }
      # if (nrow(D.t) != m) { stop("dimension of D.t must be (m * k * T) or (m * k)") }
      
      # Check time-varying and perform reshaping as needed:
      if (is.na(dim(A.t)[3]) || is.null(dim(A.t)[3]) || dim(A.t)[3] == 1) {
            A.t <- array(1, c(1, 1, time.length)) %x% matrix(A.t, nrow = m, ncol = 1)
      } 
      if (is.na(dim(B.t)[3]) || is.null(dim(B.t)[3]) || dim(B.t)[3] == 1) {
            B.t <- array(1, c(1, 1, time.length)) %x% B.t
      } 
      if (is.na(dim(C.t)[4]) || is.null(dim(C.t)[4]) || dim(C.t)[4] == 1) {
            C.t <- array(1, c(1, 1, 1, time.length)) %x% C.t
      }
      if (is.na(dim(D.t)[3]) || is.null(dim(D.t)[3]) || dim(D.t)[3] == 1) {
            D.t <- array(1, c(1, 1, time.length)) %x% D.t
      }     
      
      # Commented out final time-varying dimension check:
      # if(dim(A.t)[3] != time.length ||
      #    dim(B.t)[3] != time.length ||
      #    dim(C.t)[4] != time.length ||
      #    dim(D.t)[3] != time.length) {
      #       stop("if measurement parameters are time-varying, third dimension for A.t, B.t, D.t, and fourth dimension for C.t must be equal to the time length.")
      # }
      
      # Commented out missing, null or inf checks:
      # if(sum(is.na(A.t)) != 0 || sum(is.null(A.t)) != 0 || sum(A.t == Inf) != 0 ||
      #    sum(is.na(B.t)) != 0 || sum(is.null(B.t)) != 0 || sum(B.t == Inf) != 0 ||
      #    sum(is.na(C.t)) != 0 || sum(is.null(C.t)) != 0 || sum(C.t == Inf) != 0 ||
      #    sum(is.na(D.t)) != 0 || sum(is.null(D.t)) != 0 || sum(D.t == Inf) != 0) {
      #       stop("Check values of measurement parameters: no Inf, Null, or NA allowed")
      # }
      
      result <- list(A.t = A.t, 
                     B.t = B.t,
                     C.t = C.t,
                     D.t = D.t,
                     m = m,
                     time.length = time.length)
}











