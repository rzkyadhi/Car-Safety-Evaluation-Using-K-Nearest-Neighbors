import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("../Dataset/car.data")

label_encoder = preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data['buying']))
maint = label_encoder.fit_transform(list(data['maint']))
door = label_encoder.fit_transform(list(data['door']))
persons = label_encoder.fit_transform(list(data['persons']))
lug_boot = label_encoder.fit_transform(list(data['lug_boot']))
safety = label_encoder.fit_transform(list(data['safety']))
class_car = label_encoder.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_car)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for n in range(len(predicted)):
    print("Predicted : ", names[predicted[n]], "Data : ", x_test[n], "Actual : ", names[y_test[n]])