from RunTimeEncoder import RunTimeEncoder
from RunTimeVocoder import RunTimeVocoder

import numpy as np
from scipy.io.wavfile import write as wav_write

import json
import optparse
import sys

from flask import Flask, request, send_file
app = Flask(__name__)


@app.route('/healthcheck', methods=['GET'])
def health_check():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/synthesis', methods=['GET'])
def get_wav():
    out_file = 'out.wav'
    data = json.loads(request.data.decode('utf-8'), encoding='utf-8')

    try:
        text = data['text']
    except:
        return json.dumps({'error': 'text not set'}), 400, {'ContentType': 'application/json'}

    spect = np.asanyarray([k.npvalue() for k in encoder.predict(text)])
    signal = vocoder.synthesize(spect)
    wav_write(out_file, 24000, signal)

    return send_file(out_file, mimetype='audio/wav')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--host', action='store', dest='host',  default='0.0.0.0',
                      help='Host IP of the WebService (default="0.0.0.0")')
    parser.add_option('--port', action='store', dest='port', type='int', default=8080,
                      help='Port IP of the WebService (default=8080)')
    params, _ = parser.parse_args(sys.argv)

    encoder = RunTimeEncoder()
    encoder.load('models')

    vocoder = RunTimeVocoder()
    vocoder.load('models')

    app.run(host=params.host, port=params.port)
