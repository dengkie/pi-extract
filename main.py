import os
import feature

data_dir = './data/'

while True:
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        f = feature.extract_wav_feature(file_path)
        print(f)
