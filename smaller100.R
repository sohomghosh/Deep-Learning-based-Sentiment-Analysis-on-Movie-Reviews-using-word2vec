setwd("C:\\Users\\Administrator\\Desktop\\DB")
#library(data.table)


(NOT DO) data<-read.csv("feature_set_file.csv",header=F, stringsAsFactors = T)
(NOT DO) write.csv(file="smaller100_feature_set_file.csv",data[1:100,])
data<-read.csv("smaller100_feature_set_file.csv",header=T, stringsAsFactors = T)
data<-na.omit(data)
indep<-as.matrix(data[,2:2001])
dep<-data[,2002]


##########################Cross_validation_approach########################################
library(xgboost)
seed <- 235 #This step may be avoided
set.seed(seed) #This step may be avoided
model_xgb_cv <- xgb.cv(data=indep, label=as.matrix(dep),missing = NaN, objective="binary:logistic", nfold=10, nrounds=1200, eta=0.02, max_depth=5, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="rmse")
write.csv(file="model_xgb_crossValidation.csv",capture.output(model_xgb_cv)) 
write.csv(file="model_xgb_crossValidation_summary.csv",summary(model_xgb_cv))
# cross-validation
#model_xgb_cv <- xgb.cv(data=indep, label=as.matrix(dep), objective="binary:logistic", nfold=2, nrounds=1200, eta=0.02, max_depth=5, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="auc")


# load the library
library(caret)

# define training control
train_control <- trainControl(method="cv", number=10)
# fix the parameters of the algorithm
#grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
#model <- train(Species~., data=iris, trControl=train_control, method="nb", tuneGrid=grid)
# summarize results
model <- train(data[,2002]~., data=as.matrix(data), trControl=train_control, method="rf")
#print(model)
write.csv(file="model_rf.csv",capture.output(model))

model1 <- train(data[,2002]~., data=as.matrix(data), trControl=train_control, method="svmRadialSigma")
write.csv(file="model_svm.csv",capture.output(model1))

model2 <- train(data[,2002]~., data=as.matrix(data), trControl=train_control, method="dnn")
write.csv(file="model_dnn.csv",capture.output(model2))

model3 <- train(data[,2002]~., data=as.matrix(data), trControl=train_control, method="logicBag")

write.csv(file="model_lr.csv",capture.output(model3))



##########################Train_test_split_approach########################################
## 75% of the sample size
smp_size <- floor(0.66 * 100)

## set the seed to make your partition reproductible
set.seed(123) #This step may be avoided
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- as.matrix(data[train_ind, ])
test <- as.matrix(data[-train_ind, ])

library(randomForest)
model_rf<-randomForest(train[,2:2001],train[,2002])
model_rf_pred<-predict(model_rf,test[,2:2001])
#rf_split<-confusionMatrix(model_rf_pred, test[,2002])
rf_split<-sqrt(mean((model_rf_pred-test[,2002])**2))
write.csv(file="rf_split_pred_actual.csv",cbind(model_rf_pred,test[,2002]))
write.csv(file="rf_split.csv",rf_split)
model_rf_pred[model_rf_pred>.5]=1
model_rf_pred[model_rf_pred<=.5]=0
write.csv(file="rf_split_pred_actual0_1.csv",cbind(model_rf_pred,test[,2002]))
rf_split<-sqrt(mean((model_rf_pred-test[,2002])**2))
write.csv(file="rf_split0_1.csv",rf_split)


library(e1071)
model_svm<-svm(train[,2:2001],train[,2002])
model_svm_pred<-predict(model_svm,test[,2:2001])
svm_split<-sqrt(mean((model_svm_pred-test[,2002])**2))
write.csv(file="svm_split.csv",svm_split)
write.csv(file="svm_split_pred_actual.csv",cbind(model_svm_pred,test[,2002]))
model_svm_pred[model_svm_pred>.5]=1
model_svm_pred[model_svm_pred<=.5]=0
write.csv(file="svm_split_pred_actual0_1.csv",cbind(model_svm_pred,test[,2002]))
svm_split<-sqrt(mean((model_svm_pred-test[,2002])**2))
write.csv(file="svm_split0_1.csv",svm_split)


library(deepnet)
x <- as.matrix(train[,2:2001])
y <- as.numeric(train[,2002])
nn <- dbn.dnn.train(x,y,hidden = c(10), activationfun = "sigm",numepochs = 300,learningrate = 0.1,momentum = 0.5)
model_dnn_pred <-nn.predict(nn,test[,2:2001])
dnn_split<-sqrt(mean((model_dnn_pred-test[,2002])**2))
write.csv(file="dnn_split.csv",dnn_split)
write.csv(file="dnn_split_pred_actual.csv",cbind(model_dnn_pred,test[,2002]))
model_dnn_pred[model_dnn_pred>.5]=1
model_dnn_pred[model_dnn_pred<=.5]=0
write.csv(file="dnn_split_pred_actual0_1.csv",cbind(model_dnn_pred,test[,2002]))
dnn_split<-sqrt(mean((model_dnn_pred-test[,2002])**2))
write.csv(file="dnn_split0_1.csv",dnn_split)


library(xgboost)
bst <- xgboost(data = train[,2:2001], label = train[,2002], max.depth = 6,
               eta = .4, nthread = 2, nround = 15)
pred <- predict(bst, test[,2:2001])
xg_split<-sqrt(mean((pred-test[,2002])**2))
write.csv(file="xg_split.csv",xg_split)
write.csv(file="xg_split_pred_actual.csv",cbind(pred,test[,2002]))
pred[pred>.5]=1
pred[pred<=.5]=0
write.csv(file="xg_split_pred_actual0_1.csv",cbind(pred,test[,2002]))
xg_split<-sqrt(mean((pred-test[,2002])**2))
write.csv(file="xg_split0_1.csv",xg_split)

####ENSEMBLE#####
# x is a matrix of multiple predictions coming from different learners
#y is a vector of all output flags
#x_test is a matrix of multiple predictions on an unseen sample
prediction_table<-cbind(predict(model_rf,train[,2:2001]),predict(model_svm,train[,2:2001]),nn.predict(nn,train[,2:2001]),predict(bst, train[,2:2001]))
x <- as.matrix(prediction_table)
y <- as.numeric(train[,2002])
nn <- dbn.dnn.train(x,y,hidden = c(10),
                    activationfun = "sigm",numepochs = 300,learningrate = 0.1,momentum = 0.5)
prediction_table<-cbind(model_rf_pred,model_svm_pred,model_dnn_pred,pred)
model_ensemble <-nn.predict(nn,prediction_table)
ensemble<-sqrt(mean((model_ensemble-test[,2002])**2))
write.csv(file="ensemble.csv",ensemble)
write.csv(file="ensemble_split_pred_actual.csv",cbind(model_ensemble,test[,2002]))
model_ensemble[model_ensemble>.5]=1
model_ensemble[model_ensemble<=.5]=0
write.csv(file="ensemble_split_pred_actual0_1.csv",cbind(model_ensemble,test[,2002]))
ensemble<-sqrt(mean((model_ensemble-test[,2002])**2))
write.csv(file="ensemble_split0_1.csv",ensemble)




#----------------------------THE END----------------------------------#


'''






Manually



train_df_no_na= predictors of train = indep
train[,12] = reponse of train = dep
test_df_no_na = predictors od test =
test_response = expected response that are given of the test set =
test_output = actual test output got from running the model on test set =


model_rf<-randomForest(train_df_no_na,train[,12])
model_rf_pred<-predict(model_rf,test_df_no_na)


library(e1071)
model_svm<-svm(train_df_no_na,train[,12])
model_svm_pred<-predict(model_svm,test_df_no_na)
#soln<-cbind(test[,'Item_Identifier'],test[,'Outlet_Identifier'],data.frame(model_svm_pred))
write.csv(soln,file="submission_svm.csv")


prediction_table<-train_df_no_na
library(deepnet)
x <- as.matrix(prediction_table)
y <- as.numeric(train[,12])
nn <- dbn.dnn.train(x,y,hidden = c(10),
activationfun = "sigm",numepochs = 300,learningrate = 0.1,momentum = 0.5)

prediction_table<-test_df_no_na
x_test<- as.matrix(prediction_table)

nn_predict_test <- nn.predict(nn,x_test)

#soln<-cbind(test[,'Item_Identifier'],test[,'Outlet_Identifier'],data.frame(nn_predict_test))
write.csv(soln,file="submission_DNN.csv")

#Only deep net 2802.07550613

library(xgboost)
bst <- xgboost(data = as.matrix(train_df_no_na), label = train[,12], max.depth = 6,
eta = .4, nthread = 2, nround = 15)
pred <- predict(bst, as.matrix(test_df_no_na))
#Item_Identifier <- test[,'Item_Identifier']
#Outlet_Identifier<- test[,'Outlet_Identifier']
Item_Outlet_Sales<-data.frame(pred)
soln<-cbind(Item_Identifier,Outlet_Identifier,Item_Outlet_Sales)
write.csv(soln,file="submission_XGBoost3.csv",row.names=FALSE)
#XG boost-3 reloaded as above 1169.35465908
#Only XG boost -2   1194.74521024


model_rf_pred_train<-predict(model_rf,train_df_no_na)
model_svm_pred_train<-predict(model_svm,train_df_no_na)
prediction_table<-cbind(model_rf_pred_train,model_svm_pred_train)
library(deepnet)
x <- as.matrix(prediction_table)
y <- as.numeric(train[,12])
nn <- dbn.dnn.train(x,y,hidden = c(10),
activationfun = "sigm",numepochs = 300,learningrate = 0.1,momentum = 0.5)

prediction_table<-cbind(model_rf_pred,model_svm_pred)
x_test<- as.matrix(prediction_table)

nn_predict_test <- nn.predict(nn,x_test)

#soln<-cbind(test[,'Item_Identifier'],test[,'Outlet_Identifier'],data.frame(nn_predict_test))
write.csv(soln,file="submission_ensembleDNN___RFandSVM.csv")


splitdf <- function(dataframe, seed=NULL) {
if (!is.null(seed)) set.seed(seed)
index <- 1:nrow(dataframe)
trainindex <- sample(index, trunc(length(index)/5))
trainset <- dataframe[trainindex, ]
testset <- dataframe[-trainindex, ]
list(trainset=trainset,testset=testset)
}


'''