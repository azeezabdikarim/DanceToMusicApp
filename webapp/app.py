from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sys
import werkzeug
from werkzeug.utils import secure_filename

current_directory = os.getcwd()
models_dir = os.path.join(current_directory, '..')
sys.path.append(models_dir)

from processing.video_processor import process_video


app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'} #{'mp4', 'avi', 'mov'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper to check allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath)
            file.save(filepath)

            # Process the video and get the path to the processed video
            processed_video_path = process_video(filepath)

            # Redirect to a new route that serves the processed video
            return redirect(url_for('download_file', filename=os.path.basename(processed_video_path)))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def download_file(filename):
    return render_template('download.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

