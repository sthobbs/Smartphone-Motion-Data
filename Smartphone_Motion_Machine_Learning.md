#### *Steven Hobbs* {.author}

#### *Friday, July 24, 2015* {.date}

### Summary

In this analysis we apply machine learning ensemble methods to
smartphone motion data to classify physical activities. The analysis
consists of three parts. (I) Getting and cleaning the raw data, (II)
choosing a model, and (III) evaluating that model. I end up selecting a
model which combines stochastic gradient boosting, random forests,
linear discriminant analysis and boosted logistic regression to train
the data. Using less than 14% of the training set, this model correctly
classifies physical activities in the rest of the training set with
0.9668874 accuracy. Surprisingly, when we use the whole training set we
only have 0.9540028 accuracy, which correspondes to a 4.59972% out of
sample error.

### Raw Data

The data is from the Human Activity Recognition Using Smartphones
Dataset Version 1.0

“The experiments have been carried out with a group of 30 volunteers
within an age bracket of 19-48 years. Each person performed six
activities (WALKING, WALKING\_UPSTAIRS, WALKING\_DOWNSTAIRS, SITTING,
STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the
waist. Using its embedded accelerometer and gyroscope, we captured
3-axial linear acceleration and 3-axial angular velocity at a constant
rate of 50Hz. The experiments have been video-recorded to label the data
manually. The obtained dataset has been randomly partitioned into two
sets, where 70% of the volunteers was selected for generating the
training data and 30% the test data.” For more information see the
README file in the data set.

### Getting and Cleaning the Data

``` {.r}
# Download the data
temp<-tempfile() # Create a temporary file
fileURL<-"http://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"
download.file(fileURL,destfile=temp) # Download file
files<-unzip(temp) # Unzip files into working directory
unlink(temp) # Remove temp file

# Read in the data
activity_labels<-read.table(files[1])
features<-read.table(files[2])
subject_test<-read.table(files[14])
X_test<-read.table(files[15])
y_test<-read.table(files[16])
subject_train<-read.table(files[26])
X_train<-read.table(files[27])
y_train<-read.table(files[28])

# Create test set
for (i in 1:nrow(subject_test)){ # Subject and activity columns
    subject_test[i,2]<-activity_labels$V2[y_test[i,1]]
}
names(subject_test)<-c("subject","activity")
names(X_test)<-features$V2 # Naming the columns based on features
test<-cbind(subject_test,X_test) # combining test data

# Create training set
for (i in 1:nrow(subject_train)){ # Subject and activity columns
    subject_train[i,2]<-activity_labels$V2[y_train[i,1]]
}
names(subject_train)<-c("subject","activity")
names(X_train)<-features$V2 # Naming the columns based on features
train<-cbind(subject_train,X_train) # Combining test data

# Rename columns with identical names by adding "_x" to the end of duplicates, where x is a number.
for (col_name in unique(names(train))){
    if(sum(names(train)==col_name)>1){ # If there is >1 column with this name
        indices<-which(names(train)==col_name)
        for (i in 2:length(indices)){
            names(train)[indices[i]]<-paste0(names(train)[indices[i]],"_",i)
            names(test)[indices[i]]<-paste0(names(test)[indices[i]],"_",i)
        }
    }
}

# Load packages
library(caret)
```

### Choosing a Model

``` {.r}
# Split train data into subsets training and testing
set.seed(1832)
training_size<-1000 # training set size (<7352)
inTraining<-sample(1:nrow(train),training_size)
training<-train[inTraining,]
testing<-train[-inTraining,] 
```

#### Model 1: Stochastic Gradient Boosting

``` {.r}
ptm <- proc.time()
fit1<-train(activity~.,data=training,method="gbm",verbose=FALSE)
proc.time() - ptm
```

    ##    user  system elapsed 
    ## 1435.08    1.42 1436.75

``` {.r}
pred1<-predict(fit1,testing)
confusionMatrix(pred1,testing$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9529282

#### Model 2: Random Forests

``` {.r}
ptm <- proc.time()
fit2<-train(activity~.,data=training,method="rf",trControl=trainControl(repeats=25))
proc.time() - ptm
```

    ##    user  system elapsed 
    ## 1312.68    2.26 1314.95

``` {.r}
pred2<-predict(fit2,testing)
confusionMatrix(pred2,testing$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9467884

#### Model 3: Linear Discriminant Analysis

``` {.r}
ptm <- proc.time()
suppressWarnings(fit3<-train(activity~.,data=training,method="lda"))
proc.time() - ptm
```

    ##    user  system elapsed 
    ##   48.43    1.09   49.53

``` {.r}
pred3<-predict(fit3,testing)
confusionMatrix(pred3,testing$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9593829

#### Model 4: Boosted Logistic Regression

``` {.r}
ptm <- proc.time()
fit4<-train(activity~.,data=training,method="LogitBoost")
proc.time() - ptm
```

    ##    user  system elapsed 
    ##  374.99    0.42  375.44

``` {.r}
pred4<-predict(fit4,testing)
confusionMatrix(pred4,testing$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9508965

#### Model 5: Combined Model

``` {.r}
# Combine the predictions based on majority rule. In the event of a tie, priority is given to models which
# performed best in initial testing.
combined_pred<-NULL
for (i in 1:length(pred1)){
    preds<-c(pred3[i],pred1[i],pred4[i],pred2[i])
    num_preds<-c(sum(preds==1),sum(preds==2),sum(preds==3),sum(preds==4),sum(preds==5),sum(preds==6))
    combined_pred<-c(combined_pred,levels(pred1)[which(num_preds==max(num_preds))[1]])    
}
combined_pred<-as.factor(combined_pred)
confusionMatrix(combined_pred,testing$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9668874

We select the combined model since it gives the most accurate
classification.

### Model Evaluation

``` {.r}
set.seed(1832)

ptm <- proc.time()
fit1<-train(activity~.,data=train,method="gbm",verbose=FALSE)
pred1<-predict(fit1,test)
fit2<-train(activity~.,data=train,method="rf",trControl=trainControl(repeats=25))
pred2<-predict(fit2,test)
suppressWarnings(fit3<-train(activity~.,data=train,method="lda"))
pred3<-predict(fit3,test)
fit4<-train(activity~.,data=train,method="LogitBoost")
pred4<-predict(fit4,test)
combined_pred<-NULL
for (i in 1:length(pred1)){
    preds<-c(pred3[i],pred1[i],pred4[i],pred2[i])
    num_preds<-c(sum(preds==1),sum(preds==2),sum(preds==3),sum(preds==4),sum(preds==5),sum(preds==6))
    combined_pred<-c(combined_pred,levels(pred1)[which(num_preds==max(num_preds))[1]])    
}
combined_pred<-as.factor(combined_pred)
proc.time() - ptm
```

    ##     user   system  elapsed 
    ## 40110.22    35.09 40153.12

``` {.r}
confusionMatrix(combined_pred,test$activity)$overall[1]
```

    ##  Accuracy 
    ## 0.9540028
