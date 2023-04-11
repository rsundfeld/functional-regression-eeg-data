require(dplyr)
require(plyr)
require(Cairo)

#folds_ids <- rep(1:5, ceiling(561/5))[1:561]
folds_ids <- rep(1:5, ceiling(416/5))[1:416]

set.seed(10); folds_ids <- sample(folds_ids)
fold_iter = 5
x_train = x[which(folds_ids != fold_iter),]; y_train = y[which(folds_ids != fold_iter)]
x_test =  x[which(folds_ids == fold_iter),]; y_test =  y[which(folds_ids == fold_iter)]
#x_train  = x; y_train = y;

# functional predictors (similar to Tutz & Gertheiss (2010))
simfx1 <- function(n, p, tps, varx=rep(0,p), bx=5, mx=2*pi)
{
  fx <- list()
  fobs <- list()
  for (j in 1:p)
  {
    tmax <- max(tps[[j]])
    fx[[j]] <- matrix(0, n, length(tps[[j]]))
    fobs[[j]] <- matrix(0, n, length(tps[[j]]))
    for (i in 1:n)
    {
      bij <- runif(5, 0, bx)
      mij <- runif(5, 0, mx)
      tfx <- function(tp)
      {
        (sum(bij*sin(tp*(5-bij)*(2*pi/tmax)) - mij) + 15)/100
      }
      fx[[j]][i,] <- sapply(tps[[j]],tfx)
    }
  }
  fx <- lapply(fx, scale)
  for (j in 1:p)
  {
    fx[[j]] <- fx[[j]]/10
    for (i in 1:n)
    {
      fobs[[j]][i,] <- fx[[j]][i,] + rnorm(length(tps[[j]]), 0, sqrt(varx[j]))
    }
  }
  return(list("funx"=fx, "funcs"=fobs))
}

#set.seed(123)
tps <- list()
for(i in 1:channels) {
  tps[[i]] <- 1:time_points
}
#tps[[1]] <- tps[[2]] <- tps[[3]] <- tps[[4]] <- tps[[5]] <- tps[[6]] <- 1:time_points
n <- nrow(x_train)#trials_train
XX <- simfx1(n = n, p = channels, tps = tps, varx = rep(0,channels))

# true coefficient functions
#tp <- 1:300
#bt <- 5*dgamma(tp/10, 3, 1/3)
#plot(tp, bt, type="l")

# y values
#sigma2 <- 4
#mu <- XX$funx[[1]]%*%bt# + 0.4*XX$funx[[1]]%*%bt
#set.seed(42)
#y <- mu + rnorm(n, 0, sqrt(sigma2))
#median(y)
#mediana é -0.04 ou -0.46
#y <- ifelse(y < median(y), 1, 0)
y_train %>% table
y_train %>% class

y_train <- as.matrix(y_train)
y_train %>% class

XX_list <- list()
for(i in 1:channels) {
  XX_list[[i]] <- fda.usc::fdata.cen(x_train[,i+1])$Xcen$data
}

# functional grpl function
source("grplfunct.r")

# fit using functional smooth group lasso
lambda <- 10^seq(2,0,by=-1)
phi <- 10^c(10,8,6,4,2)
grpl1 <- grplFlogit(Y = y_train, X = XX_list, Tps = tps, lambda = lambda, phi = phi[1],
                    dfs = 10)
grpl2 <- grplFlogit(Y = y_train, X = XX_list, Tps = tps, lambda = lambda, phi = phi[2],
                    dfs = 10)
grpl3 <- grplFlogit(Y = y_train, X = XX_list, Tps = tps, lambda = lambda, phi = phi[3],
                    dfs = 10)
grpl4 <- grplFlogit(Y = y_train, X = XX_list, Tps = tps, lambda = lambda, phi = phi[4],
                    dfs = 10)
grpl5 <- grplFlogit(Y = y_train, X = XX_list, Tps = tps, lambda = lambda, phi = phi[5],
                    dfs = 10)

lasso_coefs_lambda1 <- lasso_coefs_lambda2 <- lasso_coefs_lambda3 <- matrix(0, nrow = channels, ncol = length(phi))
for(i in 1:channels) {
  lasso_coefs_lambda1[i,] = c(grpl1$Coef[[i]][,1] %>% abs %>% mean()*100, grpl2$Coef[[i]][,1] %>% abs %>% mean()*100, 
                              grpl3$Coef[[i]][,1] %>% abs %>% mean()*100, grpl4$Coef[[i]][,1] %>% abs %>% mean()*100, grpl5$Coef[[i]][,1] %>% abs %>% mean()*100)
}
for(i in 1:channels) {
  lasso_coefs_lambda2[i,] = c(grpl1$Coef[[i]][,2] %>% abs %>% mean()*100, grpl2$Coef[[i]][,2] %>% abs %>% mean()*100, 
                              grpl3$Coef[[i]][,2] %>% abs %>% mean()*100, grpl4$Coef[[i]][,2] %>% abs %>% mean()*100, grpl5$Coef[[i]][,2] %>% abs %>% mean()*100)
}
for(i in 1:channels) {
  lasso_coefs_lambda3[i,] = c(grpl1$Coef[[i]][,3] %>% abs %>% mean()*100, grpl2$Coef[[i]][,3] %>% abs %>% mean()*100, 
                              grpl3$Coef[[i]][,3] %>% abs %>% mean()*100, grpl4$Coef[[i]][,3] %>% abs %>% mean()*100, grpl5$Coef[[i]][,3] %>% abs %>% mean()*100)
}

apply(lasso_coefs_lambda2, 1, FUN=sum) %>% round(2)
#resultados desse apply:

#x_train original da base(316 linhas): fingmov: 27, 28, 25, 26, 17, 20
#[1] 0.37 1.11 1.24 0.26 0.14 0.63 1.63 0.48 0.57 0.41 0.68 1.36 0.14 0.31
#[x] 0.53 0.23 2.00 0.92 1.36 1.72 0.66 0.70 1.56 1.70 3.13 2.13 4.96 4.56

#x_train sendo a base completa (416 linhas): fingmov: 27, 28, 25, 26, 17, 7
# [1] 0.52 1.89 1.27 0.42 0.10 0.54 2.36 1.15 0.72 0.65 0.64 1.42 0.14 0.32
# [x] 0.26 0.17 2.51 1.24 1.35 2.22 1.00 0.78 1.50 1.83 3.06 2.90 6.00 5.12

#fingmov iter1: 27, 28, 25, 26, 2, 7
# [1] 0.35 2.36 1.20 0.81 0.14 0.71 1.85 0.81 0.67 0.51 0.33 1.66 0.16 0.46 
# [x] 0.36 0.53 2.23 0.89 1.64 1.76 0.63 0.83 1.29 1.49 2.76 2.76 6.19 4.70

#fingmov iter2: 27, 28, 25, 26, 7, 17
# [1] 0.42 1.21 0.95 0.39 0.14 0.88 2.85 1.25 0.70 0.63 0.42 1.21 0.19 0.27
# [x] 0.06 0.45 2.71 1.02 1.52 1.91 0.51 0.14 0.99 1.67 3.56 2.23 4.74 4.59

#fingmov iter3: 27, 28, 25, 26, 7, 17
# [1] 0.70 1.97 1.39 0.49 0.13 0.30 2.21 0.80 0.90 0.47 0.69 1.30 0.13 0.15 
# [x] 0.27 0.01 2.09 1.03 0.98 1.95 0.86 0.80 1.80 1.38 2.60 2.42 4.51 4.26

#fingmov iter4: 27, 28, 25, 26, 17, 20
# [1] 0.47 1.32 1.30 0.29 0.24 0.38 1.70 0.75 0.60 0.61 0.75 1.07 0.34 0.20 
# [x] 0.39 0.08 2.07 1.07 1.29 1.97 0.98 0.90 1.59 1.93 2.29 2.14 4.85 4.72

#fingmov iter5: 27, 28, 25, 26, 7, 17
# [1] 0.62 1.62 1.03 0.66 0.02 0.41 2.16 1.14 0.64 0.40 0.80 0.83 0.18 0.53
# [x] 0.09 0.15 2.06 1.23 1.44 1.81 0.61 0.73 1.40 1.68 3.04 2.04 5.54 4.96



#x_train original da base (268 linhas): selfreg: 4 channels with biggest coefficients on average are 1, 2, 5 and 6.
#[1] 0.42 0.08 0.01 0.07 0.08 0.26
#x_train sendo a base completa (561): selfreg: channels 1,4,5,6
#[1] 0.488 0.095 0.011 0.097 0.109 0.260
#selfreg iter1: channels 1,4,5,6
#[1] 0.53 0.11 0.03 0.14 0.13 0.31
#selfreg iter2: channels 1,2,5,6
#[1] 0.46 0.11 0.01 0.07 0.10 0.29
#selfreg iter3: channels 1,2,5,6
#[1] 0.46 0.09 0.03 0.05 0.12 0.29
#selfreg iter4: channels 1,4,5,6
#[1] 0.486 0.076 0.005 0.079 0.116 0.241
#selfreg iter5: channels  1,4,5,6
#[1] 0.51 0.07 0.01 0.11 0.09 0.22






#graphs below
Cairo::Cairo(
  30, #length
  30, #width
  file = paste("figura_exemplo_grplasso_mini", ".png", sep = ""),
  type = "png", #tiff
  bg = "transparent", #white or transparent depending on your requirement 
  dpi = 600,
  units = "cm" #you can change to pixels etc 
)

for(i in 0:floor(channels/3)){
  par(mfrow = c(3,3))

  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(1+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl3$Coef[[1+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^6))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(1+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl4$Coef[[1+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^4))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(1+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl5$Coef[[1+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^2))
  
  

  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(2+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl3$Coef[[2+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^6))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(2+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl4$Coef[[2+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^4))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(2+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl5$Coef[[2+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^2))
  
  
  

  plot(1, type="n", xlab="t", ylab=bquote(beta[.(3+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl3$Coef[[3+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^6))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(3+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl4$Coef[[3+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^4))
  
  plot(1, type="n", xlab="t", ylab=bquote(beta[.(3+i*3)](t)), xlim=c(0, time_points), ylim=c(-0.01,0.01), xaxt="n"); axis(1, at = 10*1:5); abline(h = 0)
  for(wl in 1:length(lambda))
    lines(1:time_points,grpl5$Coef[[3+i*3]][,wl],col=wl+1)
  title(expression(varphi==10^2))
}

dev.off()