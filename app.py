from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file from the form
    file = request.files['image']

    # Read image from file object
    image = Image.open(file)
    image = image.resize((450, 250))
    image_arr = np.array(image)

    # Convert to grayscale
    grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Apply dilation
    dilated = cv2.dilate(blur, np.ones((3, 3)))

    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Use CascadeClassifier for car detection
    car_cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    # Draw rectangles around each detected car and count
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert annotated image back to PIL format
    annotated_image = Image.fromarray(image_arr)

    # Convert PIL image to bytes for web display
    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Encode image to base64 for HTML embedding
    img_base64 = b64encode(img_byte_arr).decode('utf-8')

    return render_template('index.html', image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
