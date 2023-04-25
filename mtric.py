import numpy as np

# Define the metrics
def precision(y_true, y_pred):
    return np.mean((y_true == y_pred) & (y_pred != 0)) / np.mean(y_pred != 0)

def recall(y_true, y_pred):
    return np.mean((y_true == y_pred) & (y_pred != 0)) / np.mean(y_true != 0)

def f1_score(y_true, y_pred):
    return 2 * precision * recall / (precision + recall)

# # Load the image
# img = Image.open(‘image.jpg’)
#
# # Convert the image to a numpy array
# img_array = np.array(img)
#
# # Preprocess the image
# # …
#
# # Predict the class labels
# y_pred = model.predict(img_array)

# Compute the metrics
precision = precision(y_true, y_pred)
recall = recall(y_true, y_pred)
f1_score = f1_score(y_true, y_pred)

print(“Precision: {:.2f}%”.format(