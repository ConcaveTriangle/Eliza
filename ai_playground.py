from scipy.io import wavfile
import noisereduce

rate, data = wavfile.read("./audio_samples/mary/Mary-Recording-41.wav")
print("Original rate:", rate)

reduce_noise = noisereduce.reduce_noise(y=data, sr=rate)
wavfile.write("reduced_noise.wav", rate, reduce_noise)
