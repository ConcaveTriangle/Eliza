from scipy.io import wavfile
import noisereduce
import numpy as np

for x in range(77):
    x += 1
    path = "./mary/Mary-Recording-" + str(x) + ".wav"
    print("Starting: " + path)
    rate, data = wavfile.read(path)
    original_shape = data.shape
    data = np.reshape(data, (2, -1))

    reduce_noise = noisereduce.reduce_noise(y=data, sr=rate, prop_decrease=0.95)
    wavfile.write("testing-" + str(x) + ".wav" , rate, reduce_noise.reshape(original_shape))
