# Machine Learning for Roads Segmentation

# Project Structure:
The project is divided into three modules:

1. filters.py
This module contains functions for extracting image features using various filters and transformations. It includes operations like Gabor filters, Canny Edge detection, Hough Transform, Roberts Edge detection, Sobel, Scharr, Prewitt, Gaussian filtering, and Median filtering.

2. train.py
The train.py script demonstrates the training process using a RandomForestClassifier. It uses the features extracted by filters.dataframe for both the input image and a masked image. The model is trained, and its accuracy is evaluated. The trained model is then saved for later use.

3. test.py
The test.py script loads a pre-trained RandomForestClassifier model from the saved file (roads_model). It applies the model to a new test image, and the resulting segmentation is visualized using a color map.
