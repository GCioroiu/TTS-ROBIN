import dynet_config
dynet_config.set(mem=2048, random_seed=9)

import dynet as dy
import numpy as np


class RunTimeEncoder:
    def __init__(self):
        self.model = dy.Model()

        self.phone_lookup = self.model.add_lookup_parameters((42, 100))
        self.feature_lookup = self.model.add_lookup_parameters((3, 100))
        self.speaker_lookup = self.model.add_lookup_parameters((18, 200))

        self.encoder_fw = dy.VanillaLSTMBuilder(1, 100, 256, self.model)
        self.encoder_bw = dy.VanillaLSTMBuilder(1, 100, 256, self.model)
        self.decoder = dy.VanillaLSTMBuilder(2, 812, 1024, self.model)

        self.hid_w = self.model.add_parameters((500, 1024))
        self.hid_b = self.model.add_parameters((500,))

        self.proj_w_1 = self.model.add_parameters((80, 500))
        self.proj_b_1 = self.model.add_parameters((80,))

        self.proj_w_2 = self.model.add_parameters((80, 500))
        self.proj_b_2 = self.model.add_parameters((80,))

        self.proj_w_3 = self.model.add_parameters((80, 500))
        self.proj_b_3 = self.model.add_parameters((80,))

        self.highway_w = self.model.add_parameters((80, 712))

        self.last_mgc_proj_w = self.model.add_parameters((100, 80))
        self.last_mgc_proj_b = self.model.add_parameters((100,))

        self.stop_w = self.model.add_parameters((1, 1024))
        self.stop_b = self.model.add_parameters((1,))

        self.att_w1 = self.model.add_parameters((100, 712))
        self.att_w2 = self.model.add_parameters((100, 1024))
        self.att_v = self.model.add_parameters((1, 100))

        self.start_lookup = self.model.add_lookup_parameters((1, 80))
        self.decoder_start_lookup = self.model.add_lookup_parameters((1, 812))

        self.char2int = {}

    def _make_input(self, text):
        x = [self.phone_lookup[self.char2int['START']]]
        for c in text:
            c = c.lower()
            char_emb = self.phone_lookup[self.char2int[c]]

            if c == c.upper():
                char_emb += self.feature_lookup[1]
            else:
                char_emb += self.feature_lookup[0]

            x.append(char_emb)
        x.append(self.phone_lookup[self.char2int['STOP']])
        return x

    def _get_speaker_embedding(self):
        return self.speaker_lookup[12]

    def predict(self, text):
        dy.renew_cg()

        output_mgc = []
        last_mgc = self.start_lookup[0]

        x = self._make_input(text)
        x_speaker = self._get_speaker_embedding()

        x_fw = self.encoder_fw.initial_state().transduce(x)
        x_bw = self.encoder_bw.initial_state().transduce(reversed(x))

        encoder = [dy.concatenate([fw, bw, x_speaker]) for fw, bw in zip(x_fw, reversed(x_bw))]
        encoder = dy.concatenate(encoder, 1)
        hidden_encoder = self.att_w1 * encoder

        decoder = self.decoder.initial_state().add_input(self.decoder_start_lookup[0])

        last_att_pos = 0
        warm_up = 5
        finish = 5

        for k in range(5 * len(text)):
            attention_weights = (self.att_v * dy.tanh(hidden_encoder + self.att_w2 * decoder.output()))[0]
            current_pos = np.argmax(attention_weights.npvalue())

            if current_pos < last_att_pos:
                current_pos = last_att_pos

            if current_pos > last_att_pos:
                current_pos = last_att_pos + 1

            last_att_pos = current_pos
            att = dy.select_cols(encoder, [current_pos])

            if warm_up > 0:
                last_att_pos = 0
                warm_up -= 1

            if last_att_pos >= len(text):
                if finish > 0:
                    finish -= 1
                else:
                    break

            mgc_proj = dy.tanh(self.last_mgc_proj_w * last_mgc + self.last_mgc_proj_b)
            decoder = decoder.add_input(dy.concatenate([mgc_proj, att]))
            hidden = dy.tanh(self.hid_w * decoder.output() + self.hid_b)

            highway_hidden = self.highway_w * att
            output_mgc.append(dy.logistic(highway_hidden + self.proj_w_1 * hidden + self.proj_b_1))
            output_mgc.append(dy.logistic(highway_hidden + self.proj_w_2 * hidden + self.proj_b_2))
            output_mgc.append(dy.logistic(highway_hidden + self.proj_w_3 * hidden + self.proj_b_3))

            last_mgc = output_mgc[-1]

            '''
                Stop layer seems to finish in about 40% of cases with an average of 3 steps earlier
                However, it introduces another matmul, and further testing proved that it is faster without it

                The testing was done on a sentence level, the results may be better on a word parallelization, so 
                I'm not removing it yet.
            '''
            # output_stop = dy.tanh(self.stop_w * decoder.output() + self.stop_b)
            # if output_stop.value() < -0.5:
            #     break

        return output_mgc

    def load(self, path):
        self.model.populate('%s/rnn_encoder.network' % path)

        with open('%s/char2int' % path, 'rt', encoding='utf-8') as f:
            for line in f.readlines():
                c, i = line.split('\t')
                self.char2int[c] = int(i)
