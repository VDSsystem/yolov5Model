from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import subprocess

app = Flask(__name__)
cors = CORS(app, resources={r'*': {'origins': '*'}})
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        id = request.json['id']
        url = f"https://vadss.vercel.app/api/savedImages?id={id}"
        response = requests.get(url)
        data = response.json()
        url = data['url']
        # Call the detect.py script with the image URL as an argument
        command = f"python ./yolov5/detect.py --source {url} --weights ./best.pt"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        resp = jsonify({'output': output.decode('utf-8')})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp
    else:
        return "YOLOv5 Model APP"
