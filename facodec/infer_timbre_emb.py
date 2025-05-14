import os
import json
import torch
import torchaudio
from tqdm import tqdm 
from ns3_codec import FACodecEncoder, FACodecDecoder

# 加载配置文件
config_path = 'configs/VADV_baseline.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 从配置文件获取 facodec 配置
facodec_config = config['facodec']

# 获取路径
audio_dir = facodec_config['audio_dir']
encoder_ckpt = facodec_config['encoder_ckpt']
decoder_ckpt = facodec_config['decoder_ckpt']
spk_embs_save_base_path = facodec_config['spk_embs_save_base_path']

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移动到 GPU
fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
).to(device)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
).to(device)

# 加载模型权重
fa_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
fa_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))

fa_encoder.eval()
fa_decoder.eval()

# 使用 torchaudio 加载音频文件
def load_audio_torchaudio(wav_path):
    try:
        wav, sample_rate = torchaudio.load(wav_path)
        # 将音频重采样到 16kHz
        wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
        wav = wav.unsqueeze(0).to(device)  # 将音频数据移动到 GPU
        return wav
    except Exception as e:
        print(f"Failed to load audio {wav_path}: {e}")
        return None

# 获取所有子文件夹中的音频文件
def get_all_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

# 加载音频文件目录
audio_files = get_all_wav_files(audio_dir)

# 使用 tqdm 进度条来跟踪音频文件的处理进度
with torch.no_grad():
    for audio_file in tqdm(audio_files, desc="Processing audio files", ncols=100):
        # 加载音频文件
        test_wav = load_audio_torchaudio(audio_file)
        if test_wav is None:
            print(f"加载{test_wav}失败\n")
            continue  # 如果加载失败，跳过

        # Encode音频
        enc_out = fa_encoder(test_wav)

        # Decode并获取spk_embs
        _, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

        # 获取音频文件名称，去掉路径和后缀 .wav
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]

        # 提取 speaker_id（假设 speaker_id 是文件名中的前缀）
        speaker_id = audio_name.split('_')[0]

        # 构造保存路径
        speaker_folder = os.path.join(spk_embs_save_base_path, speaker_id)
        os.makedirs(speaker_folder, exist_ok=True)

        # 构造保存文件路径
        save_path = os.path.join(speaker_folder, f'{audio_name}.pt')

        # 保存数据
        torch.save(spk_embs.cpu(), save_path)

print("所有音频处理完毕，speaker embeddings 已保存。")


