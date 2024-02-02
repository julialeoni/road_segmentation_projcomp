import cv2
import pandas as pd
import filters
 
img = cv2.imread('test_img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.reshape(-1)

df = filters.dataframe(img)

masked_img = cv2.imread('mask_img.png')
masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
masked_img1 = masked_img.reshape(-1)
df['Masks'] = masked_img1

Y = df["Masks"].values
X = df.drop(["Masks"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state = 42)
model.fit(X_train, Y_train)

prediction_test_train = model.predict(X_train)
prediction_test = model.predict(X_test)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(Y_train, prediction_test_train))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

import pickle
filename = "roads_model"
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)
segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap ='jet')
plt.axis('off')

