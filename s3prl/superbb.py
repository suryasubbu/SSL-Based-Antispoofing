import torchaudio
from s3prl.nn import S3PRLUpstream
import torch

# Load the model
model = S3PRLUpstream("byol_s_resnetish34")
model.eval()

# Load the audio file
file_path = "/home/suryasss/output.wav"
waveform, sample_rate = torchaudio.load(file_path)

# Ensure the waveform has the appropriate shape (batch, samples)
if waveform.ndimension() == 1:
    waveform = waveform.unsqueeze(0)

# Truncate or pad to a uniform length (e.g., 16000 * 2 samples for 2 seconds)
max_length = 16000 * 10
wavs = torch.zeros(waveform.size(0), max_length)
for i, wav in enumerate(waveform):
    wavs[i, :min(max_length, wav.size(0))] = wav[:max_length]

# Create a tensor of wave lengths
wavs_len = torch.LongTensor([min(max_length, waveform.size(1)) for _ in range(waveform.size(0))])

# Run through the model
with torch.no_grad():
    all_hs, all_hs_len = model(wavs, wavs_len)