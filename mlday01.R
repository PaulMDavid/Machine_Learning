d=data.frame(iris)
d
library(caret)
library(e1071)
fit<-train(Species~.,data=d,method="knn")
x<-predict(fit,d[1,1:4])
x


y<-createDataPartition(d$Species, times = 1, p = 0.8, list = FALSE)
y
traindat<-d[y, ]
testdata<-d[-y,]
testdata
traindat
fit<-train(Species~.,data = traindat,method="knn")
predict(fit,testdata)

Ecol<-read.table(file.choose())
v<-createDataPartition(Ecol$V9,times=1,p=0.8,list=FALSE)
c
v
trainset<-Ecol[v,]
testdat<-Ecol[-v,]
fits<-train(V9~.,data = trainset,method="knn")
p=predict(fits,testdat)
confusionMatrix(p,testdat$V9)




df = read.table(file.choose())
df
boxplot(df$V2)
plot(df)
v<-createDataPartition(df$V1,times=1,p=0.8,list=FALSE)
trainset<-df[v,]
testdat<-df[-v,]
fits<-train(V4~.,data = trainset,method="knn")
p=predict(fits,testdat)
confusionMatrix(p,testdat$V4)
