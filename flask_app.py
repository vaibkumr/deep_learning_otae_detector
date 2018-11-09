from flask import Flask, jsonify, request, send_file, render_template
from flask_api import status
from flask_cors import CORS
import os
import requests
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from otae_classifier import OtaeSanDetector

UPLOAD_FOLDER = '/home/timetraveller/Documents/Projects/shimura otae classifier/images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


@app.route("/",methods=['GET'])
def index():
    return "<h1> Working </h1>"


@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "nani?"
        else:
          f = request.files['file']
          type = secure_filename(f.filename).split('.')[1]
          if type not in ALLOWED_EXTENSIONS:
              return "invalid"
          if f :
              filename = os.path.join(
                                    app.config['UPLOAD_FOLDER'],
                                    f"temp.{type}"
                                    # secure_filename(f.filename)
                                    )
              f.save(filename)
              value = OtaeSanDetector(filename).predict()
              if value == 'otae':
                  value = "Otae san detected!"
              else:
                  value = "No Otae san here"
              return render_template(
                                    "done.html",
                                    user_image = filename,
                                    result = value
                                    )


if __name__ == '__main__':
    #remove debug=True before actual deployment smh
   app.run(debug = True)
