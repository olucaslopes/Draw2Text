import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from time import strftime


def save_png(img_):
    img = Image.fromarray(img_, mode='RGBA')
    img.save('./img/img_' + strftime("run_%Y_%m_%d_%H_%M_%S") + '.png')


def plot_numbers(img_array, true_label=None, predicted_label=None, n_rows=3, n_cols=10, title=''):
    """
    Display a grid of images using matplotlib.

    Args:
    - img_array (numpy.ndarray): An array of images.
    - true_label (list, optional): A list of true labels for the images. Default is None.
    - predicted_label (list, optional): A list of predicted labels for the images. Default is None.
    - n_rows (int, optional): The number of rows in the grid. Default is 3.
    - n_cols (int, optional): The number of columns in the grid. Default is 10.
    - title (str, optional): The title of the plot. Default is an empty string.

    Returns:
    None.
    """
    if not isinstance(img_array, np.ndarray):
        raise ValueError('img_array need to be an numpy array')
    # if img_array.shape[0] < n_cols:
    #     n_cols = img_array.shape[0]
    # if img_array.shape[0] < 10:
    #     n_rows = 1
    min_color = 0
    max_color = 255
    fig = plt.figure(figsize=(1.2* n_cols, 1.2*n_rows))
    if title:
        plt.suptitle(title, weight='bold', size=18)
    for col in range(n_cols):
        for row in range(n_rows):
            index = row + max(col, 0) * n_rows
            if index >= img_array.shape[0]:
                break
            plt.subplot(n_rows, n_cols, index + 1)
            
            plt.imshow(
                img_array[index],
                cmap='binary',
                vmin=min_color, vmax=max_color,
                interpolation="nearest")

            plt.axis('off')
            
            if predicted_label is not None:
                if true_label is not None:
                    plt.title(f'label={true_label[index]}\npredicted={predicted_label[index]}')
                else:
                    plt.title(f'predicted={predicted_label[index]}')
            elif true_label is not None:
                plt.title(f'label={true_label[index]}')
    plt.tight_layout()

    return fig


def predict_drawings(img_):
    # Open image and convert to array ranging from 0-255
    pil_image = Image.fromarray(img_, mode='RGBA')
    gray_img = ImageOps.grayscale(pil_image)
    img_array = np.array(gray_img)

    # Apply thresholding to convert the grayscale image to a binary image
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on size and aspect ratio
    digits = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # if w >= 10 and h >= 20 and w / h >= 0.5:
        # TODO: change resize to maintaing aspect ratio
        digit = thresh[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28))
        digits.append(digit)

    digits = np.array(digits)

    model = tf.keras.models.load_model('models/mnist_keras_model')

    predicted_labels = np.argmax(model.predict(digits), axis=1)

    return plot_numbers(digits, predicted_label=predicted_labels)