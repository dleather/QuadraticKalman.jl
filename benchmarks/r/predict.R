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


# DESCRIPTION : This is the function for the prediction step 
# for the latent factors and the observable.
#-----------------------------------------------------------
predict.mean <- function(intercept, multi, mean) { intercept + multi %*% mean }
predict.vcov <- function(multi, vcov, condi.vcov) { multi %*% vcov %*% t(multi) + condi.vcov }



