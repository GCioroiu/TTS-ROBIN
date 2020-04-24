from RunTimeEncoder import RunTimeEncoder
from RunTimeVocoder import RunTimeVocoder

import numpy as np
from scipy.io.wavfile import write as wav_write

def main():
    rte = RunTimeEncoder()
    rte.load('models')

    rtv = RunTimeVocoder()
    rtv.load('models')

    with open('texts.txt', 'rt', encoding='utf-8') as f:
        lines = f.readlines()

        for k in range(len(lines)):
            text = lines[k].strip()

            spect = np.asanyarray([k.npvalue() for k in rte.predict(text)])

            signal = rtv.synthesize(spect)

            wav_write('tests/test_%d.wav' % k, 24000, signal)

if __name__ == '__main__':
    main()