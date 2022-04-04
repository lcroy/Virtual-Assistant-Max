from flask import Flask, abort, request, jsonify, render_template, send_file, send_from_directory, redirect, url_for
import json
import os
from playsound import playsound

app = Flask(__name__)
app.add_url_rule('/photos/<path:filename>', ...)

content = ''

@app.route('/get_conv/', methods=['GET'])
def get_conv():
    global content
    filename = os.path.join(app.static_folder, 'data', 'conv.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    if (content != str(data[0]['service'])):
        playsound('static/audio/static_sounds_fade.mp3')
        content = str(data[0]['service'])

    return jsonify(data)


@app.route('/', methods=['GET'])
def index():
    playsound('static/audio/static_sounds_fade.mp3')
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
