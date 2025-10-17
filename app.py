# app.py

from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
from detect import detect_objects

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'test_data'
app.config['RESULT_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    output_path = None
    detections = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            ext = file.filename.split('.')[-1]
            uid = str(uuid.uuid4())
            input_filename = f"{uid}.{ext}"
            output_filename = f"result_{uid}.{ext}"

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

            file.save(input_path)

            # Run detection
            detections = detect_objects(input_path, result_path)

            # For displaying images in browser
            image_path = '/' + input_path
            output_path = '/' + result_path

    return render_template('index.html',
                           image_path=image_path,
                           output_path=output_path,
                           detections=detections)

# Serve uploaded images
@app.route('/test_data/<filename>')
def uploaded_file(filename):
    return app.send_static_file(os.path.join('test_data', filename))

@app.route('/results/<filename>')
def result_file(filename):
    return app.send_static_file(os.path.join('results', filename))

if __name__ == '__main__':
    app.run(debug=True)
