# project
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.
## Program:
## I)Perform ROI from an image
```python
# Import necessary packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert the image into RGB
image_path = 'TOM&JERRY.jpeg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Create an ROI mask
roi_mask = np.zeros_like(img_rgb)
roi_mask[100:250, 100:250, :] = 255  # Define the ROI region

# Segment the ROI using bitwise AND operation
segmented_roi = cv2.bitwise_and(img_rgb, roi_mask)

# Display the segmented ROI
plt.imshow(segmented_roi)
plt.title('Segmented ROI')
plt.axis('off')
plt.show()
```
## II)Perform handwritting detection in an image
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'handwriting.jpg'
detect_handwriting(image_path)
```
## III)Perform object detection with label in an image
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'handwriting.jpg'
detect_handwriting(image_path)
```
## OUTPUT :

### I)Perform ROI from an image:

![Screenshot 2025-05-28 132413](https://github.com/user-attachments/assets/b72da26a-2d45-4ff5-ae68-3d7895db91f9)

![Screenshot 2025-05-28 132423](https://github.com/user-attachments/assets/814ec307-f92d-4fe5-93f2-dc015ca48032)
/assets/9f3f6ce2-d44c-45b7-99a0-6c99438b9b4f)

### II)Perform handwritting detection in an image:

![Screenshot 2025-05-28 132727](https://github.com/user-attachments/assets/341d73f4-65b3-420b-852c-a62fa256bb69)

### III)Perform object detection with label in an image:

![Screenshot 2025-05-28 133034](https://github.com/user-attachments/assets/19898697-8fcc-4052-9b32-84ff77a1d55f)

## Result:
Thus, a python program using OpenCV for following image manipulations is done successfully
