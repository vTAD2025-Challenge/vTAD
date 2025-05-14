import os
import json
import torch
import librosa
from glob import glob
from tqdm import tqdm
import torchaudio.compliance.kaldi as kaldi
from model.ecapa_tdnn_xvector import ECAPA_TDNN
import yaml

# Load JSON configuration file
with open('configs/VADV_baseline.json', 'r') as f:
    config_data = json.load(f)

# Extract necessary paths from the JSON config
audio_dir = config_data["ecapa_tdnn"]["audio_dir"]
params_path = config_data["ecapa_tdnn"]["params"]
yaml_path = config_data["ecapa_tdnn"]["yaml"]
outpath = config_data["ecapa_tdnn"]["outpath"]

# Load the ECAPA model state
with open(params_path, "rb") as f:
    ecapa_state = torch.load(f, map_location=torch.device("cpu"))

model_params = {
    "inputs_dim": 40,
    "num_targets": 7205,
    "aug_dropout": 0., "tail_dropout": 0.,
    "training": False, "extracted_embedding": "near",
    "ecapa_params": {"channels": 1024,
                     "embd_dim": 192,
                     "mfa_conv": 1536,
                     "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}},

    "pooling": "ecpa-attentive",
    "pooling_params": {
        "hidden_size": 128,
        "time_attention": True,
        "stddev": True,
    },
    "fc1": False,
    "fc1_params": {
        "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
    "fc2_params": {
        "nonlinearity": '', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},
    "margin_loss": True,
    "margin_loss_params": {
        "method": "aam", "m": 0.2, "feature_normalize": True,
        "s": 30, "mhe_loss": False, "mhe_w": 0.01},
    "use_step": False,
    "step_params": {
        "margin_warm": False,
        "margin_warm_conf": {"start_epoch": 1, "end_epoch": 1, "offset_margin": -0.0, "init_lambda": 1.0},
        "T": None,
        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
        "s": False, "s_tuple": (30, 12), "s_list": None,
        "t": False, "t_tuple": (0.5, 1.2),
        "p": False, "p_tuple": (0.5, 0.1)}
}

class rawWav2KaldiFeature(object):
    def __init__(self, feature_type='fbank', kaldi_featset={}, mean_var_conf={}):
        super().__init__()
        assert feature_type in ['mfcc', 'fbank']
        self.feat_type = feature_type
        self.kaldi_featset = kaldi_featset
        if self.feat_type == 'mfcc':
            self.extract = kaldi.mfcc
        else:
            self.extract = kaldi.fbank
        if mean_var_conf is not None:
            self.mean_var = InputSequenceNormalization(**mean_var_conf)
        else:
            self.mean_var = torch.nn.Identity()

    def __call__(self, wav, wav_len, sample_rate=16000):
        self.kaldi_featset['sample_frequency'] = sample_rate
        lens = wav_len
        waveforms = wav * (1 << 15)
        feats = []
        for i, waveform in enumerate(waveforms):
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
            waveform = waveform[:, :lens[i].long()]
            feat = self.extract(waveform, **self.kaldi_featset)
            feat = self.mean_var(feat)
            feat = feat.transpose(-1, -2)
            feats.append(feat)
        feats, _ = batch_pad_right(feats)
        return feats


use_gpu = True
torch.cuda.set_device(1)
model = ECAPA_TDNN(**model_params)
model.eval()
model.load_state_dict(ecapa_state, strict=False)
if use_gpu:
    model.cuda()

# Load YAML configuration for feature extraction setup
with open(yaml_path, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Feature extraction setup
from libs.egs.kaldi_features import InputSequenceNormalization
from libs.support.utils import batch_pad_right

num_iter = 10
alpha = 0.00002
epsilon = 0.002
decay_factor = 1.0
extract_fbank = rawWav2KaldiFeature(**cfg["feature_extraction_conf"])

def gen_adv_wav_glob(path_glob, outpath):
    for file_names in tqdm(path_glob, dynamic_ncols=True):
        if ".wav" in file_names:
            waveform_x, sr = librosa.load(file_names, sr=16000)
            
            with torch.no_grad():
                x = torch.tensor(waveform_x).unsqueeze(0).float()
                x_len = torch.Tensor([x.shape[-1]])
                if use_gpu:
                    x = x.cuda()
                    x_len = x_len.cuda()
                
                ori_fbank_y = extract_fbank(x, x_len)
                speaker_embedding = model(ori_fbank_y).squeeze(-1).cpu()
                
                subfolder_name = file_names.split(os.path.sep)[-1].split(".")[0].split("_")[0]
                subfolder_path = os.path.join(outpath, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)
                
                file_base_name = os.path.basename(file_names).split(".")[0]
                save_path = os.path.join(subfolder_path, f"{file_base_name}.pt")
                torch.save(speaker_embedding, save_path)

                torch.cuda.empty_cache()
                del x, x_len, ori_fbank_y, speaker_embedding

def get_all_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

if __name__ == '__main__':
    input_file_paths = get_all_wav_files(audio_dir)
    gen_adv_wav_glob(input_file_paths, outpath)
