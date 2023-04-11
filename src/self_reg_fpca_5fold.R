require(fda.usc)
require(ggplot2)
require(reshape)
require(grid)
require(glmnet)
library(randomForest)
library(caTools)

#para cada iter de 1 a 5, o rol de variáveis selecionadas (código lasso_selection) muda:
#selfreg iter1: channels 1,4,5,6
#selfreg iter2: channels 1,2,5,6
#selfreg iter3: channels 1,2,5,6
#selfreg iter4: channels 1,4,5,6
#selfreg iter5: channels 1,4,5,6
 
setwd("C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Self Regulation\\")
ajustar_modelos = F

#0. creating folds for 5fold analysis

folds_ids <- rep(1:5, ceiling(561/5))[1:561]
set.seed(10); folds_ids <- sample(folds_ids)

#funcreg_acc <- funcreg_roc <- lasso_acc <- lasso_roc <- rforest_acc <- rforest_roc <- vector()
funcreg_total_predictions <- lasso_total_predictions <- rforest_total_predictions <- y_test_total <- vector()


### ITERATIONS 2,3
for(fold_iter in 2:3) {
  print(Sys.time())
  print(fold_iter)
  x_train = x[which(folds_ids != fold_iter),]; y_train = y[which(folds_ids != fold_iter)]
  x_test =  x[which(folds_ids == fold_iter),]; y_test =  y[which(folds_ids == fold_iter)]
  y_test_total <- append(y_test_total, y_test)
  
  
  #1. transform the features to "Functional Data" type
  x.v1.fdata <- fdata(x_train$V01) 
  x.v2.fdata <- fdata(x_train$V02)
  x.v3.fdata <- fdata(x_train$V05)
  x.v4.fdata <- fdata(x_train$V06)
  
  x.v1.fdata <- fdata.cen(x.v1.fdata)$Xcen; x.v2.fdata <- fdata.cen(x.v2.fdata)$Xcen; x.v3.fdata <- fdata.cen(x.v3.fdata)$Xcen; x.v4.fdata <- fdata.cen(x.v4.fdata)$Xcen; 
  
  #2. Quantos PC's são necessários para explicar 90% da variabilidade
  #summary(fdata2pc(x.v1.fdata, ncomp = 2))
  #summary(fdata2pc(x.v2.fdata, ncomp = 2))
  #summary(fdata2pc(x.v3.fdata, ncomp = 2))
  #summary(fdata2pc(x.v4.fdata, ncomp = 2))
  
  if(ajustar_modelos) {
    #3. Implementa um cross-validation com K folds para achar o nº de PC's
    k = 5
    n_features = 4
    folds <- rep(1:k, ceiling(length(y_train)/k))[1:length(y_train)]
    x_formula_list <- number.pcs.basis_x1_list <- number.pcs.basis_x2_list <- number.pcs.basis_x3_list <- number.pcs.basis_x4_list <- mean_acc <- std_acc <- vector()
    vetor_temp_duplic <- data.frame(matrix(vector(), 0, 5))
    
    for(number.pcs.basis_x1 in 1:2){
      print(number.pcs.basis_x1)
      for(number.pcs.basis_x2 in 1:2){
        for(number.pcs.basis_x3 in 1:2){
          for(number.pcs.basis_x4 in 1:2){
            features_grid <- rep(list(0:1), n_features) %>% expand.grid()
            colnames(features_grid) <- c('x1', 'x2', 'x3', 'x4')
            for(i in 2:(2^n_features)){
              x_formula = noquote(paste('`y_train[-testIndexes]` ~', paste(names(features_grid[which(features_grid[i,]==1,arr.ind=T)[,2]]), collapse = '+')))
              acc = NULL
              
              #checa a duplicidade
              basis_vector <- c(number.pcs.basis_x1, number.pcs.basis_x2, number.pcs.basis_x3, number.pcs.basis_x4)
              features_loop <- ifelse(c('x1', 'x2', 'x3', 'x4') %in% paste(names(features_grid[which(features_grid[i,]==1,arr.ind=T)[,2]])), basis_vector, 0)
              nrow = nrow(rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop)))))
              nrow_unique = nrow(distinct(rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop))))))
              if (nrow == nrow_unique) {
                vetor_temp_duplic <- rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop))))
                for(j in 1:k){
                  testIndexes <- which(folds==j,arr.ind=TRUE)
                  
                  basis.x1=create.pc.basis(x.v1.fdata[-testIndexes,], l = 1:number.pcs.basis_x1)
                  basis.x2=create.pc.basis(x.v2.fdata[-testIndexes,], l = 1:number.pcs.basis_x2)
                  basis.x3=create.pc.basis(x.v3.fdata[-testIndexes,], l = 1:number.pcs.basis_x3)
                  basis.x4=create.pc.basis(x.v4.fdata[-testIndexes,], l = 1:number.pcs.basis_x4)
                  
                  basis.x=list("x1"=basis.x1, "x2"=basis.x2, "x3"=basis.x3, "x4"=basis.x4)
                  
                  ldata.train.cv=list("df"=as.data.frame(y_train[-testIndexes]), 
                                      "x1"=x.v1.fdata[-testIndexes,], 
                                      "x2"=x.v2.fdata[-testIndexes,], 
                                      "x3"=x.v3.fdata[-testIndexes,], 
                                      "x4"=x.v4.fdata[-testIndexes,])
                  
                  ldata.valid.cv=list("df"=as.data.frame(y_train[testIndexes]), 
                                      "x1"=x.v1.fdata[testIndexes,], 
                                      "x2"=x.v2.fdata[testIndexes,], 
                                      "x3"=x.v3.fdata[testIndexes,], 
                                      "x4"=x.v4.fdata[testIndexes,])
                  
                  res.basis.cv=fregre.glm(as.formula(x_formula), ldata.train.cv,family=binomial(),basis.x=basis.x)    
                  
                  table.cv = table(y_train[testIndexes], ifelse(predict(res.basis.cv, newx = ldata.valid.cv) < 0.5, 0, 1))
                  acc = append(acc, table.cv %>% diag() %>% sum () / table.cv %>% sum())
                }
                x_formula_list <- append(x_formula_list, x_formula)
                mean_acc <- append(mean_acc, mean(acc))
                std_acc <- append(std_acc, sd(acc))
                number.pcs.basis_x1_list <- append(number.pcs.basis_x1_list, number.pcs.basis_x1)
                number.pcs.basis_x2_list <- append(number.pcs.basis_x2_list, number.pcs.basis_x2)
                number.pcs.basis_x3_list <- append(number.pcs.basis_x3_list, number.pcs.basis_x3)
                number.pcs.basis_x4_list <- append(number.pcs.basis_x4_list, number.pcs.basis_x4)
              } 
            }
          }
        }
      }
    }
    
    beta_cv_matrix <- as.data.frame(cbind(x_formula_list, number.pcs.basis_x1_list, number.pcs.basis_x2_list, number.pcs.basis_x3_list, number.pcs.basis_x4_list, mean_acc, std_acc))
    colnames(beta_cv_matrix) <- c('formula', 'b1', 'b2', 'b3', 'b4', 'mean_acc', 'std_acc')
    best_row <- beta_cv_matrix[which.max(beta_cv_matrix$mean_acc),]
    best_row
    #write.csv2(beta_cv_matrix, "beta_cv_matrix_fingmov_fourier.csv")
    
    #4. Cria as bases Spline/Fourier
    ldata.train=list("df"=as.data.frame(y_train),"x1"=x.v1.fdata,"x2" =x.v2.fdata,
                     "x3"=x.v3.fdata,"x4" =x.v4.fdata)
    basis.pc1=create.pc.basis(x.v1.fdata, l = 1:as.numeric(best_row$b1))
    basis.pc2=create.pc.basis(x.v2.fdata, l = 1:as.numeric(best_row$b2))
    basis.pc3=create.pc.basis(x.v3.fdata, l = 1:as.numeric(best_row$b3))
    basis.pc4=create.pc.basis(x.v4.fdata, l = 1:as.numeric(best_row$b4))
    basis.x.pc=list("x1"=basis.pc1,"x2"=basis.pc2,"x3"=basis.pc3,"x4"=basis.pc4)            
    
    final_formula <- noquote(paste('y_train ~', strsplit(best_row$formula, '~ ')[[1]][2]))
    res.pc=fregre.glm(formula = as.formula(final_formula)
                      ,ldata.train,family=binomial,basis.x=basis.x.pc)  
    saveRDS(res.pc, paste('SelfReg2 - FPCA - Fold', fold_iter,'.rds'))
  }  
  #5. measure performance metrics
  
  ldata.test=list("df"=as.data.frame(y_test), 
                  "x1"=fdata(x_test$V01) - func.mean(fdata(x_train$V01)),
                  "x2"=fdata(x_test$V02) - func.mean(fdata(x_train$V02)),
                  "x3"=fdata(x_test$V05) - func.mean(fdata(x_train$V05)),
                  "x4"=fdata(x_test$V06) - func.mean(fdata(x_train$V06)))
  
  funcreg_model <- readRDS(paste('SelfReg2 - FPCA - Fold', fold_iter,'.rds'))
  funcreg_predictions <- predict(funcreg_model, 
                                 newx = ldata.test)
  funcreg_total_predictions <- append(funcreg_total_predictions, funcreg_predictions)
  #funcreg_matrix <- table(y_test, ifelse(funcreg_predictions < 0.5, 0, 1)) %>% confusionMatrix
  #funcreg_acc <- append(funcreg_acc, funcreg_matrix$overall[[1]])
  #funcreg_roc <- append(funcreg_roc, auc(roc(y_test, funcreg_predictions))[[1]])
}



### ITERATIONS 1,4,5
for(fold_iter in c(1,4,5)) {
  print(Sys.time())
  print(fold_iter)
  x_train = x[which(folds_ids != fold_iter),]; y_train = y[which(folds_ids != fold_iter)]
  x_test =  x[which(folds_ids == fold_iter),]; y_test =  y[which(folds_ids == fold_iter)]
  y_test_total <- append(y_test_total, y_test)
  
  
  #1. transform the features to "Functional Data" type
  x.v1.fdata <- fdata(x_train$V01) 
  x.v2.fdata <- fdata(x_train$V04)
  x.v3.fdata <- fdata(x_train$V05)
  x.v4.fdata <- fdata(x_train$V06)
  
  x.v1.fdata <- fdata.cen(x.v1.fdata)$Xcen; x.v2.fdata <- fdata.cen(x.v2.fdata)$Xcen; x.v3.fdata <- fdata.cen(x.v3.fdata)$Xcen; x.v4.fdata <- fdata.cen(x.v4.fdata)$Xcen; 
  
  #2. Quantos PC's são necessários para explicar 90% da variabilidade
  #summary(fdata2pc(x.v1.fdata, ncomp = 2))
  #summary(fdata2pc(x.v2.fdata, ncomp = 2))
  #summary(fdata2pc(x.v3.fdata, ncomp = 2))
  #summary(fdata2pc(x.v4.fdata, ncomp = 2))
  
  if(ajustar_modelos) {
    #3. Implementa um cross-validation com K folds para achar o nº de PC's
    k = 5
    n_features = 4
    folds <- rep(1:k, ceiling(length(y_train)/k))[1:length(y_train)]
    x_formula_list <- number.pcs.basis_x1_list <- number.pcs.basis_x2_list <- number.pcs.basis_x3_list <- number.pcs.basis_x4_list <- mean_acc <- std_acc <- vector()
    vetor_temp_duplic <- data.frame(matrix(vector(), 0, 5))
    
    for(number.pcs.basis_x1 in 1:2){
      print(number.pcs.basis_x1)
      for(number.pcs.basis_x2 in 1:2){
        for(number.pcs.basis_x3 in 1:2){
          for(number.pcs.basis_x4 in 1:2){
            features_grid <- rep(list(0:1), n_features) %>% expand.grid()
            colnames(features_grid) <- c('x1', 'x2', 'x3', 'x4')
            for(i in 2:(2^n_features)){
              x_formula = noquote(paste('`y_train[-testIndexes]` ~', paste(names(features_grid[which(features_grid[i,]==1,arr.ind=T)[,2]]), collapse = '+')))
              acc = NULL
              
              #checa a duplicidade
              basis_vector <- c(number.pcs.basis_x1, number.pcs.basis_x2, number.pcs.basis_x3, number.pcs.basis_x4)
              features_loop <- ifelse(c('x1', 'x2', 'x3', 'x4') %in% paste(names(features_grid[which(features_grid[i,]==1,arr.ind=T)[,2]])), basis_vector, 0)
              nrow = nrow(rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop)))))
              nrow_unique = nrow(distinct(rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop))))))
              if (nrow == nrow_unique) {
                vetor_temp_duplic <- rbind(vetor_temp_duplic, t(as.data.frame(c(x_formula, features_loop))))
                for(j in 1:k){
                  testIndexes <- which(folds==j,arr.ind=TRUE)
                  
                  basis.x1=create.pc.basis(x.v1.fdata[-testIndexes,], l = 1:number.pcs.basis_x1)
                  basis.x2=create.pc.basis(x.v2.fdata[-testIndexes,], l = 1:number.pcs.basis_x2)
                  basis.x3=create.pc.basis(x.v3.fdata[-testIndexes,], l = 1:number.pcs.basis_x3)
                  basis.x4=create.pc.basis(x.v4.fdata[-testIndexes,], l = 1:number.pcs.basis_x4)
                  
                  basis.x=list("x1"=basis.x1, "x2"=basis.x2, "x3"=basis.x3, "x4"=basis.x4)
                  
                  ldata.train.cv=list("df"=as.data.frame(y_train[-testIndexes]), 
                                      "x1"=x.v1.fdata[-testIndexes,], 
                                      "x2"=x.v2.fdata[-testIndexes,], 
                                      "x3"=x.v3.fdata[-testIndexes,], 
                                      "x4"=x.v4.fdata[-testIndexes,])
                  
                  ldata.valid.cv=list("df"=as.data.frame(y_train[testIndexes]), 
                                      "x1"=x.v1.fdata[testIndexes,], 
                                      "x2"=x.v2.fdata[testIndexes,], 
                                      "x3"=x.v3.fdata[testIndexes,], 
                                      "x4"=x.v4.fdata[testIndexes,])
                  
                  res.basis.cv=fregre.glm(as.formula(x_formula), ldata.train.cv,family=binomial(),basis.x=basis.x)    
                  
                  table.cv = table(y_train[testIndexes], ifelse(predict(res.basis.cv, newx = ldata.valid.cv) < 0.5, 0, 1))
                  acc = append(acc, table.cv %>% diag() %>% sum () / table.cv %>% sum())
                }
                x_formula_list <- append(x_formula_list, x_formula)
                mean_acc <- append(mean_acc, mean(acc))
                std_acc <- append(std_acc, sd(acc))
                number.pcs.basis_x1_list <- append(number.pcs.basis_x1_list, number.pcs.basis_x1)
                number.pcs.basis_x2_list <- append(number.pcs.basis_x2_list, number.pcs.basis_x2)
                number.pcs.basis_x3_list <- append(number.pcs.basis_x3_list, number.pcs.basis_x3)
                number.pcs.basis_x4_list <- append(number.pcs.basis_x4_list, number.pcs.basis_x4)
              } 
            }
          }
        }
      }
    }
    
    beta_cv_matrix <- as.data.frame(cbind(x_formula_list, number.pcs.basis_x1_list, number.pcs.basis_x2_list, number.pcs.basis_x3_list, number.pcs.basis_x4_list, mean_acc, std_acc))
    colnames(beta_cv_matrix) <- c('formula', 'b1', 'b2', 'b3', 'b4', 'mean_acc', 'std_acc')
    best_row <- beta_cv_matrix[which.max(beta_cv_matrix$mean_acc),]
    best_row
    #write.csv2(beta_cv_matrix, "beta_cv_matrix_fingmov_fourier.csv")
    
    #4. Cria as bases Spline/Fourier
    ldata.train=list("df"=as.data.frame(y_train),"x1"=x.v1.fdata,"x2" =x.v2.fdata,
                     "x3"=x.v3.fdata,"x4" =x.v4.fdata)
    basis.pc1=create.pc.basis(x.v1.fdata, l = 1:as.numeric(best_row$b1))
    basis.pc2=create.pc.basis(x.v2.fdata, l = 1:as.numeric(best_row$b2))
    basis.pc3=create.pc.basis(x.v3.fdata, l = 1:as.numeric(best_row$b3))
    basis.pc4=create.pc.basis(x.v4.fdata, l = 1:as.numeric(best_row$b4))
    basis.x.pc=list("x1"=basis.pc1,"x2"=basis.pc2,"x3"=basis.pc3,"x4"=basis.pc4)            
    
    final_formula <- noquote(paste('y_train ~', strsplit(best_row$formula, '~ ')[[1]][2]))
    res.pc=fregre.glm(formula = as.formula(final_formula)
                      ,ldata.train,family=binomial,basis.x=basis.x.pc)  
    saveRDS(res.pc, paste('SelfReg2 - FPCA - Fold', fold_iter,'.rds'))
  }  
  #5. measure performance metrics
  
  ldata.test=list("df"=as.data.frame(y_test), 
                  "x1"=fdata(x_test$V01) - func.mean(fdata(x_train$V01)),
                  "x2"=fdata(x_test$V04) - func.mean(fdata(x_train$V04)),
                  "x3"=fdata(x_test$V05) - func.mean(fdata(x_train$V05)),
                  "x4"=fdata(x_test$V06) - func.mean(fdata(x_train$V06)))
  
  funcreg_model <- readRDS(paste('SelfReg2 - FPCA - Fold', fold_iter,'.rds'))
  funcreg_predictions <- predict(funcreg_model, 
                                 newx = ldata.test)
  funcreg_total_predictions <- append(funcreg_total_predictions, funcreg_predictions)
  #funcreg_matrix <- table(y_test, ifelse(funcreg_predictions < 0.5, 0, 1)) %>% confusionMatrix
  #funcreg_acc <- append(funcreg_acc, funcreg_matrix$overall[[1]])
  #funcreg_roc <- append(funcreg_roc, auc(roc(y_test, funcreg_predictions))[[1]])

}
  
#metrics dataframe
funcreg_binary_predictions <- ifelse(funcreg_total_predictions < 0.5, 0, 1)

metrics_df = as.data.frame(cbind(y_test_total, 
                                 funcreg_total_predictions, funcreg_binary_predictions))
#acc
acc_funcreg <- 1 - mean(abs(metrics_df$y_test_total - metrics_df$funcreg_binary_predictions))
acc_funcreg

#acc IC
acc_funcreg - qnorm(0.975)*sqrt(sum((metrics_df$y_test_total - metrics_df$funcreg_binary_predictions)^2))/sqrt(416)/sqrt(416-1);
acc_funcreg + qnorm(0.975)*sqrt(sum((metrics_df$y_test_total - metrics_df$funcreg_binary_predictions)^2))/sqrt(416)/sqrt(416-1);

#roc auc
roc_funcreg <- auc(roc(metrics_df$y_test_total, metrics_df$funcreg_total_predictions))[[1]]
roc_funcreg