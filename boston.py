from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
boston=load_boston()
features=boston.data
target=boston.target



print (features.shape)




features_train,features_test,labels_train,labels_test=train_test_split(features,target,test_size=0.3)
print features_train.shape
print features_test.shape

reg=Lasso(alpha=0.1,normalize=False,random_state=1)
reg.fit(features_train,labels_train)
pred=reg.predict(features_test)

print(r2_score(labels_test,pred))
print(mean_absolute_error(labels_test,pred))
print(mean_squared_error(labels_test,pred))


for i in range(len(labels_test)):
	#print (labels_test[i],pred[i])
	plt.scatter(labels_test[i],pred[i])
plt.plot(labels_test,reg.predict(features_test))
plt.show()