import json
import os
import numpy as np
import argparse

def main(data_path):
    for idx in range(1, 1501):
        json_path = os.path.join(data_path, f'{idx}', f'{idx}_feature.json')
        output_path = os.path.join(data_path, f'{idx}', f'{idx}_pitch.txt')
        
        with open(json_path, "r") as f:
            feature = json.loads(f.read())
        pitch = np.array(feature['vocal_pitch'])
        energy = np.array(feature['energy'])
        zcr = np.array(feature['zcr'])
        with open(output_path, "w") as f:
            for i in range(pitch.shape[0]):
                s = '{:.6f}'.format(pitch[i])
                f.write(s+'\n')

def analysis(data_path):
    for idx in range(1, 1501):
        json_path = os.path.join(data_path, f'{idx}', f'{idx}_feature.json')
        output_path = os.path.join(data_path, f'{idx}', f'{idx}_pitch.txt')
        
        with open(json_path, "r") as f:
            feature = json.loads(f.read())
        pitch = np.array(feature['vocal_pitch'])
        energy = np.array(feature['energy'])
        zcr = np.array(feature['zcr'])
        
        onset = 0.016
        with open(output_path, "w") as f:
            for i in range(pitch.shape[0]):
                s = '{:.4f}: {:.6f}\t{:.6f}\t{:.9f}'.format(onset, pitch[i], energy[i], zcr[i])
                f.write(s+'\n')
                onset += 0.032

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./AIcup_testset_ok')
    args = parser.parse_args()
    main(args.data_path)