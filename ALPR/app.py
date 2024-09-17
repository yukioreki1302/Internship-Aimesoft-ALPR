from flask import Flask, request, render_template, redirect, url_for
import cv2
import os
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
import re
import numpy as np

# Chỉ định đường dẫn đến tesseract nếu cần
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
model = YOLO(r'C:\Users\admin\OneDrive\Documents\GitHub\Internship-Aimesoft-ALPR\ALPR\runs\detect\train4\weights\best.pt')

# Set up the path for uploaded files
UPLOAD_FOLDER = r'C:\Users\admin\OneDrive\Documents\GitHub\Internship-Aimesoft-ALPR\ALPR\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to process image
def process_image(image_path):
    results = model.predict(image_path, device='cpu')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Vẽ hình chữ nhật bao quanh biển số xe
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Lấy vùng ROI (biển số xe) để xử lý OCR
            roi = gray_image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, lang='eng', config=r'--oem 3 --psm 8')

            # Loại bỏ các ký tự không phải là chữ số hoặc chữ cái
            clean_text = re.sub(r'[^A-Za-z0-9]', '', text)

            # Hiển thị kết quả sau khi làm sạch
            cv2.putText(image, f'{clean_text}', (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 0), 2)

    # Lưu ảnh đã xử lý vào thư mục 'static/uploads/'
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Trả về đường dẫn tương đối cho HTML
    return 'uploads/result.jpg'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result_image_path = process_image(filepath)
            return render_template('index.html', result_image=result_image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
