import IPython
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave as we
import soundfile as sf
import os
import librosa
from pesq import pesq

clean_path = "/home/ap/Desktop/Workspace/DTLN/valset_clean/p232_007.wav"
denoised_path = '/home/ap/Desktop/Workspace/DTLN/output-denoised-audio (1).wav'

# denoised_files = os.listdir(denoised_path)
# clean_files = os.listdir(clean_path)

# rmse = []
# snr_list = []
# pesq_list = []

# for i in range(len(denoised_files)):
original, sr1 = librosa.load(clean_path, sr=None)
denoised, sr2 = librosa.load(denoised_path, sr=None)


min_length = min(len(original), len(denoised))
original = original[:min_length]
denoised = denoised[:min_length]

mse = np.mean((original - denoised) ** 2)

# Calculate Signal-to-Noise Ratio (SNR)
signal_power = np.mean(original ** 2)
noise_power = np.mean((original - denoised) ** 2)
snr = 10 * np.log10(signal_power / noise_power)

# rmse.append(mse)
# snr_list.append(snr)


print(f"Mean Squared Error: {np.average(mse)}")
print(f"Signal-to-Noise Ratio (SNR): {np.average(snr)} dB")

rate, ref = wavfile.read("/home/ap/Desktop/Workspace/DTLN/test/19-198-0034.wav")
rate, deg = wavfile.read("/home/ap/Desktop/Workspace/DTLN/output_test/19-198-0034.wav")

print(pesq(rate, ref, deg, 'wb'))
