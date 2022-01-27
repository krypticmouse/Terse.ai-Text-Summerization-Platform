import os
import json
import boto3

from utils import summarize
from flask import Flask, render_template, request, redirect, url_for

ACCESS_ID = ''
ACCESS_KEY = ''
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('main.html')

@app.route('/explore', methods = ['GET'])
def explore():
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    s3.download_file('summary-major-project', 'database.json', 'database.json')

    data = json.load(open('database.json'))
    
    os.remove('database.json')
    return render_template('explore.html', data = data)

@app.route('/generate', methods = ['GET'])
def generate():
    return render_template('generate.html', name = "")

@app.route('/summary', methods = ['POST'])
def summary():
    url = request.form.get("url")
    name, summary = summarize(url)
    
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    s3.download_file('summary-major-project', 'database.json', 'database.json')

    data = json.load(open('database.json'))
    
    data.append({
        'url': url,
        'name': name,
        'summary': summary
    })

    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    s3.Bucket('summary-major-project').put_object(Key='database.json', Body= json.dumps(data))

    os.remove('database.json')
    return render_template('generate.html', url = url, name = name, summary = summary)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
