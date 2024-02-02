import cv2
import pickle
from matplotlib import pyplot as plt
import filters

filename = "roads_model"
loaded_model = pickle.load(open(filename, 'rb'))

file = 'test_images/04.png'
imgt = cv2.imread(file)
img = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)

Z = filters.dataframe(img)
result = loaded_model.predict(Z)
segmented = result.reshape((img.shape))
plt.imshow(segmented, cmap ='jet')
plt.axis('off')


