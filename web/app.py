from flask import Flask, request, send_file
from model import load_model, inference
import os
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'models/road_detection_model.pth'
model = load_model(model_path).to(device)

@app.route('/')
def index():
    return send_file('web/index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    file_path = os.path.join('web', file.filename)
    file.save(file_path)
    result_path = inference(file_path, model)
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
