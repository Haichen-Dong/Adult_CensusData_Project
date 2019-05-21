library(tidyverse)
library(caret)
#library(class)
library(Rborist)
library(MASS)
library(naivebayes)
require(gridExtra)

#library(caret)
#library(matrixStats)
destfile="U:/projects/FinalPro/AdultClean.RData"
if(!file.exists(destfile)){
  dl <- tempfile()
  download.file("https://www.kaggle.com/uciml/adult-census-income/downloads/adult-census-income.zip/3", dl)
  
DownloadFile="U:/projects/FinalPro/adult.csv"

dat = read.csv(DownloadFile, header = TRUE)
#check data
str(dat)
head(dat)
dim(dat)
#some fields are not good for analysis
par(mfrow=c(1,2))
hist(dat$capital.gain ,xlab= "capital.gain",col = "yellow",border = "blue",main="Figure 1",cex.lab=.5, cex.axis=.5, cex.main=.8, cex.sub=.5)
hist(dat$capital.loss ,xlab= "capital.loss",col = "yellow",border = "blue",main="Figure 2",cex.lab=.5, cex.axis=.5, cex.main=.8, cex.sub=.5)
ggplot(dat, aes(x=native.country)) + geom_bar(colour = "red")+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+ggtitle("Figure 3")
plot(dat$income,dat$fnlwgt,main="Figure 4")
#Clean data
dat_clean<-dat[c(-3,-5,-11,-12,-14)]
dat_clean<- filter(dat_clean,workclass!="?" & occupation!="?") 

str(dat_clean)
#head(dat_clean)
any(is.na(dat_clean))
#dat_clean[duplicated(dat_clean),]

#Save cleaned file
save(dat_clean, file = "U:/projects/FinalPro/AdultClean.RData")
}
load(destfile)
#data summary
dim(dat_clean)
summary(dat_clean)
prop.table(table(dat_clean$income))
str(dat_clean)

par(mfrow=c(1,2))
p1<-plot(dat_clean$income,dat_clean$age,main="Age")
p2<-plot(dat_clean$income,dat_clean$hours.per.week,main="HourPerWeek")

g1<-ggplot(dat_clean, aes(x=workclass,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
g2<-ggplot(dat_clean, aes(x=education,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
g3<-ggplot(dat_clean, aes(x=marital.status,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
g4<-ggplot(dat_clean, aes(x=occupation,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
grid.arrange(g1, g2, g3, g4, ncol=2)
g5<-ggplot(dat_clean, aes(x=relationship ,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
g6<-ggplot(dat_clean, aes(x=race,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
g7<-ggplot(dat_clean, aes(x=sex,fill=income)) + geom_bar()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
grid.arrange(g5, g6, g7, ncol=2)

#Prepare Training/Testing Data
set.seed(1)
dat_clean$income <- ifelse(dat_clean$income == "<=50K", 0, 1)
dat_clean$income <- as.factor(dat_clean$income)
test_index <- createDataPartition(y = dat_clean$income, times = 1, p = 0.2, list = FALSE)
TrainData <- dat_clean[-test_index,]
TestData <- dat_clean[test_index,]
dim(TrainData)
dim(TestData)
prop.table(table(TrainData$income))
prop.table(table(TestData$income))


#Models
# naive bayes
fit_naive_bayes <- naive_bayes(income~., data=TrainData)
y_hat_naive_bayes <- predict(fit_naive_bayes, TestData)
cm <- confusionMatrix(y_hat_naive_bayes, TestData$income)
Predit_Accuracy <- tibble(method = "Naive Bayes",  accuracy=cm$overall["Accuracy"])
as.table(c(cm$overall["Accuracy"],cm$byClass["Sensitivity"],cm$byClass["Specificity"]))

#Logistic Regression
fit_glm <- glm(income ~., data=TrainData, family = "binomial")
p_hat_glm <- predict(fit_glm, TestData)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 1, 0))
cm <-confusionMatrix(data = y_hat_glm, reference = TestData$income)
Predit_Accuracy <- bind_rows(Predit_Accuracy,
                             tibble(method="Logistic Regression",  
                                    accuracy=cm$overall["Accuracy"]))
#Stepwise Logistic Regression 
fit_step<-fit_glm %>%stepAIC(trace=FALSE,direction="both")
#summary(fit_step)
p_hat_step<- predict(fit_step, TestData,type = "response")
y_hat_step<- factor(ifelse(p_hat_step > 0.5, 1, 0))
cm <-confusionMatrix(data = y_hat_step, reference = TestData$income)
Predit_Accuracy <- bind_rows(Predit_Accuracy,
                             tibble(method="Stepwise Logistic Regression",  
                                    accuracy=cm$overall["Accuracy"]))
#k-Nearest Neighbor
fit_knn <- knn3(income~., data=TrainData,  k = 7)
y_hat_knn <- predict(fit_knn, TestData, type="class")
cm <- confusionMatrix(y_hat_knn, TestData$income)
Predit_Accuracy <- bind_rows(Predit_Accuracy,
                             tibble(method="Nearest Neighbor",  
                                    accuracy=cm$overall["Accuracy"]))
#Random Forest
control <- trainControl(method="cv", number = 5, p = 0.8)
grid <- expand.grid(minNode = c(1) , predFixed = c(2,3,4,5))
train_rf <-  train(TrainData[,-10], TrainData$income, 
                   method = "Rborist", 
                   nTree = 50,
                   trControl = control,
                   tuneGrid = grid)
train_rf$bestTune
fit_rf <- Rborist(TrainData[,-10], TrainData$income, 
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)
y_hat_rf <-predict(fit_rf, TestData[,-10], type="class")
cm <- confusionMatrix(y_hat_rf$yPred, TestData$income)
cm$overall["Accuracy"]
Predit_Accuracy <- bind_rows(Predit_Accuracy,
                             tibble(method="Random Forest",  
                                    accuracy=cm$overall["Accuracy"]))

#Ensembles
p <- (as.numeric(y_hat_rf$yPred) + as.numeric(y_hat_step))/2
y_pred <- as.factor(ifelse(p>=1.5,1,0))
cm <- confusionMatrix(y_pred, TestData$income)
Predit_Accuracy <- bind_rows(Predit_Accuracy,
                             tibble(method="Ensembles",  
                                    accuracy=cm$overall["Accuracy"]))
Predit_Accuracy