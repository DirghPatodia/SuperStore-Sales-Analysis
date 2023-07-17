library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(e1071)
library(rpart)
library(Metrics)
library(randomForest)
train=read.csv("superstore_train.csv")
test=read.csv("superstore_test.csv")
result=read.csv("test_result.csv")

#Feature Selection Correlation Matix
df_cor=data.frame(Sales=train$Sales,Discount=train$Discount,Quantity=train$Quantity,Postal.Code=train$Postal.Code)
cor_matrix <- data.frame(cor(df_cor))
mean(cor(df_cor))
ggcorrplot(cor_matrix)

#Feature Selection RFImportance
rf_imp=randomForest(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode+Segment+State,train)
rf_impor=importance(rf_imp,type=NULL, class=NULL, scale=TRUE,)/1000000
print(rf_impor)
varImpPlot(rf_imp)

#Multiple Linear regression 
mlr=lm(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,train)
print(summary(mlr))
dfmlr=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
predmlr=predict(mlr,dfmlr,)
rmsemlr=sqrt(mean((predmlr - result$Profit)^2, na.rm=TRUE))
print(rmsemlr)
r2mlr=R2(predmlr,result$Profit)
print(r2mlr)
maemlr=mae(result$Profit, predmlr)
print(maemlr)
cat("\n")
#decesion Tree
dt=rpart(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,train)
#print(summary(dt))
dfdt=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
preddt=predict(dt,dfdt,)
rmsedt=sqrt(mean((preddt - result$Profit)^2, na.rm=TRUE))
print(rmsedt)
r2dt=R2(preddt,result$Profit)
print(r2dt)
maedt=mae(result$Profit, preddt)
print(maedt)
cat("\n")
#Random Forest
set.seed(123)
rf=randomForest(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,ntree=2000,train)
print(rf)
print(importance(rf,type=NULL, class=NULL, scale=TRUE,))
dfrf=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
predrf=predict(rf,dfrf,)
rmserf=sqrt(mean((predrf - result$Profit)^2, na.rm=TRUE))
print(rmserf)
r2rf=R2(predrf,result$Profit)
print(r2rf)
maerf=mae(result$Profit, predrf)
print(maerf)
cat("\n")
#Support Vector Regression Radial
model_svm = svm(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,train)
print(summary(model_svm,type=NULL, class=NULL, scale=TRUE,))
dfsvm=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
predsvm=predict(model_svm,dfsvm,)
rmsesvm=sqrt(mean((predsvm - result$Profit)^2, na.rm=TRUE))
print(rmsesvm)
r2svm=R2(predsvm,result$Profit)
print(r2svm)
maesvm=mae(result$Profit, predsvm)
print(maesvm)
cat("\n")

#Support Vector Regression Linear
model_svm2 = svm(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,train,kernel= "linear")
print(summary(model_svm2,type=NULL, class=NULL, scale=TRUE,))
dfsvm2=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
predsvm2=predict(model_svm2,dfsvm2,)
rmsesvm2=sqrt(mean((predsvm2 - result$Profit)^2, na.rm=TRUE))
print(rmsesvm2)
r2svm2=R2(predsvm2,result$Profit)
print(r2svm2)
maesvm2=mae(result$Profit, predsvm2)
print(maesvm2)
cat("\n")

#Support Vector Regression Polynomial
model_svm3 = svm(Profit~Sales+Discount+Category+Quantity+Region+Ship.Mode,train,kernel= "polynomial")
print(summary(model_svm3,type=NULL, class=NULL, scale=TRUE,))
dfsvm3=data.frame(Sales=test$Sales,Discount=test$Discount,Category=test$Category,Quantity=test$Quantity,Region=test$Region,Ship.Mode=test$Ship.Mode)
predsvm3=predict(model_svm3,dfsvm3,)
rmsesvm3=sqrt(mean((predsvm3 - result$Profit)^2, na.rm=TRUE))
print(rmsesvm3)
r2svm3=R2(predsvm3,result$Profit)
print(r2svm3)
maesvm3=mae(result$Profit, predsvm3)
print(maesvm3)
cat("\n")

#Plotting the graphs
Model <- c("MLR", "Decesion Tree", "Random Forest", "SVM (Radial)", "SVM (Linear)","SVM (Polynomial)")
Rmse_Value <- c(rmsemlr, rmsedt, rmserf, rmsesvm, rmsesvm2, rmsesvm3)
rmse_df <- data.frame(Model, Rmse_Value)
rmse_df[,-1] <-round(rmse_df[,-1],2)
par(mar=c(4,8,4,2))
barplot(rmse_df[,-1],
        main = "Rmse Value Comparison",
        xlab = "Rmse Value",
        names.arg = rmse_df[,-2],las=2,
        border="red",
        col="blue",
        density=10,
        horiz = TRUE
        )

Model <- c("MLR", "Decesion Tree", "Random Forest", "SVM (Radial)", "SVM (Linear)","SVM (Polynomial)")
R2_Value <- c(r2mlr, r2dt, r2rf, r2svm, r2svm2, r2svm3)
r2_df <- data.frame(Model, R2_Value)
r2_df[,-1] <-round(r2_df[,-1],2)
par(mar=c(4,8,4,2))
barplot(r2_df[,-1],
        main = "R2 Value Comparison",
        xlab = "R2 Value",
        names.arg = r2_df[,-2],las=2,
        border="black",
        col="orange",
        density=10,
        horiz = TRUE
)

Model <- c("MLR", "Decesion Tree", "Random Forest", "SVM (Radial)", "SVM (Linear)","SVM (Polynomial)") 
mae_Value <- c(maemlr, maedt, maerf, maesvm, maesvm2, maesvm3)
mae_df <- data.frame(Model, mae_Value)
mae_df[,-1] <-round(mae_df[,-1],2)
par(mar=c(4,8,4,2))
barplot(mae_df[,-1],
        main = "mae Value Comparison",
        xlab = "mae Value",
        names.arg = mae_df[,-2],las=2,
        border="brown",
        col="purple",
        density=10,
        horiz = TRUE
)
