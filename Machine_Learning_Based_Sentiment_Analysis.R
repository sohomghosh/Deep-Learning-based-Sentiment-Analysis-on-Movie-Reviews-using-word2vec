library(RTextTools)
library(e1071)

pos_tweets =  rbind(
  c('review_1', 'positive'),
  c('review_2', 'positive'),
  c('review_3', 'positive'),
  c('review_4', 'positive'),
  c('review_5', 'positive')
)

neg_tweets = rbind(
  c('review_6', 'negative'),
  c('review_7', 'negative'),
  c('review_8', 'negative'),
  c('review_9', 'negative'),
  c('review_10', 'negative')
)

test_tweets = rbind(
  c('review_11', 'positive'),
  c('review_12', 'positive'),
  c('review_13', 'negative'),
  c('review_14', 'negative'),
  c('review_15', 'negative')
)

tweets = rbind(pos_tweets, neg_tweets, test_tweets)
# build dtm
matrix= create_matrix(tweets[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=FALSE) 
# train the model
mat = as.matrix(matrix)
classifier = naiveBayes(mat[1:10,], as.factor(tweets[1:10,2]) )
# test the validity
predicted = predict(classifier, mat[11:15,])
predicted
table(tweets[11:15, 2], predicted)
recall_accuracy(tweets[11:15, 2], predicted)
# build the data to specify response variable, training set, testing set.
#When the virgin flag is set to FALSE, it indicates that all data in the training and testing sets have corresponding labels.
#When the virgin flag is set to TRUE, it indicates that the testing set is unclassified data with no known true values.
container = create_container(matrix, as.numeric(as.factor(tweets[,2])),
                             trainSize=1:10, testSize=11:15,virgin=FALSE)
#MAXENT is Maximum entropy classifier
#For more details refer: http://blog.datumbox.com/machine-learning-tutorial-the-max-entropy-text-classifier/
#A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
models = train_models(container, algorithms=c("MAXENT" , "SVM", "RF", "BAGGING", "TREE"))
results = classify_models(container, models)
# accuracy table
table(as.numeric(as.factor(tweets[11:15, 2])), results[,"FORESTS_LABEL"])
table(as.numeric(as.factor(tweets[11:15, 2])), results[,"MAXENTROPY_LABEL"])

# recall accuracy
recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[,"SVM_LABEL"])
# model summary
analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)
analytics@ensemble_summar
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")

###################
"load data"
###################
setwd("C:/Users/Sohom Ghosh/Desktop/WORK!!!!/0MathLogic/April2ndWeek/sentimentAnalysis/SentimentAnalysisMachineLearning")
happy = readLines("./happy.txt")
sad = readLines("./sad.txt")
happy_test = readLines("./happy_test.txt")
sad_test = readLines("./sad_test.txt")

tweet = c(happy, sad)
tweet_test= c(happy_test, sad_test)
tweet_all = c(tweet, tweet_test)
sentiment = c(rep("happy", length(happy) ), 
              rep("sad", length(sad)))
sentiment_test = c(rep("happy", length(happy_test) ), 
                   rep("sad", length(sad_test)))
sentiment_all = as.factor(c(sentiment, sentiment_test))

library(RTextTools)

# naive bayes
#Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. 
#http://www.tfidf.com/
mat= create_matrix(tweet_all, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

mat = as.matrix(mat)

classifier = naiveBayes(mat[1:160,], as.factor(sentiment_all[1:160]))
predicted = predict(classifier, mat[161:180,]); predicted

table(sentiment_test, predicted)
recall_accuracy(sentiment_test, predicted)

# the other methods
mat= create_matrix(tweet_all, language="english", 
                   removeStopwords=FALSE, removeNumbers=TRUE, 
                   stemWords=FALSE, tm::weightTfIdf)

container = create_container(mat, as.numeric(sentiment_all),
                             trainSize=1:160, testSize=161:180,virgin=FALSE) #????????????removeSparseTerms

models = train_models(container, algorithms=c("MAXENT",
                                              "SVM",
                                              #"GLMNET", "BOOSTING", 
                                              "SLDA","BAGGING", 
                                              "RF", # "NNET", 
                                              "TREE" 
))

# test the model
results = classify_models(container, models)
table(as.numeric(as.numeric(sentiment_all[161:180])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.numeric(sentiment_all[161:180])),results[,"FORESTS_LABEL"])

# formal tests
analytics = create_analytics(container, results)
summary(analytics)

head(analytics@algorithm_summary)
head(analytics@label_summary)
head(analytics@document_summary)
head(analytics@ensemble_summary) # Ensemble Agreement

# Cross Validation
N=3
cross_SVM = cross_validate(container,N,"SVM")
cross_MAXENT = cross_validate(container,N,"MAXENT")
#cross_GLMNET = cross_validate(container,N,"GLMNET")


