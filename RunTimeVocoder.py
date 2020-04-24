import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.distributions.normal import Normal
from clarinet.wavenet import Wavenet
from clarinet.wavenet_iaf import Wavenet_Student

class RunTimeVocoder:
    def __init__(self):
        self.model_t = Wavenet(out_channels=2,
                             num_blocks=4,
                             num_layers=6,
                             residual_channels=128,
                             gate_channels=256,
                             skip_channels=128,
                             kernel_size=3,
                             cin_channels=80,
                             upsample_scales=[16, 16])

        self.model_s = Wavenet_Student(num_blocks_student=[1, 1, 1, 4],
                                       num_layers=6, cin_channels=80)

        self.model_t.to(device)
        self.model_s.to(device)

        self.model_t.eval()
        self.model_s.eval()


    def synthesize(self, spect):
        num_samples = len(spect) * 16 * 16
        c = torch.tensor(spect.T, dtype=torch.float32).to(device)
        q_0 = Normal(c.new_zeros((1, 1, num_samples)), c.new_ones((1, 1, num_samples)))
        z = q_0.sample()
        torch.cuda.synchronize()
        with torch.no_grad():
            c_up = self.model_t.upsample(c)
            y = self.model_s.generate(z, c_up).squeeze()
        torch.cuda.synchronize()
        return y.cpu().numpy()

    def load(self, path):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                self.model_t.load_state_dict(torch.load(path + "/nn_vocoder.network", map_location='cuda:0'))
                self.model_s.load_state_dict(torch.load(path + "/pnn_vocoder.network", map_location='cuda:0'))
            else:
                self.model_t.load_state_dict(torch.load(path + "/nn_vocoder.network"))
                self.model_s.load_state_dict(torch.load(path + "/pnn_vocoder.network"))

        else:
            self.model_t.load_state_dict(torch.load(path + '/nn_vocoder.network', map_location=lambda s, l: s))
            self.model_s.load_state_dict(torch.load(path + '/pnn_vocoder.network', map_location=lambda s, l: s))

        self.model_t.to(device)
        self.model_s.to(device)
