import cv2
import numpy as np
import pytesseract
from PIL import Image


def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -15)

    # Write the image after applying opencv...
    cv2.imwrite("thresh.png", img)

    # Recognize text with pytesseract
    result = pytesseract.image_to_string(Image.open("thresh.png"))

    return result


print('--- Starting Text Recognition---')
print(get_string("3.png"))
print("------- Done -------")
