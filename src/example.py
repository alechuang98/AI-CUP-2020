
import argparse
import os
import sys
import librosa
import mido
import json
import numpy as np
from scipy import stats
import time

from madmom.audio.filters import LogarithmicFilterbank
from madmom.features.onsets import SpectralOnsetProcessor
from madmom.audio.signal import normalize
from scipy import signal

class Note:
    def __init__(self, frame, frame_pitch, onset_time, offset_time):
        self.frame_pitch = frame_pitch
        self.frame = frame
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.pitch = 0

def get_onset(wav_path):

    y, sr = librosa.core.load(wav_path, sr= None)
    sos = signal.butter(25, 100, btype= 'highpass', fs= sr, output='sos')
    wav_data= signal.sosfilt(sos, y)
    wav_data= normalize(wav_data)

    sodf = SpectralOnsetProcessor(onset_method='complex_flux', fps= 50, filterbank=LogarithmicFilterbank, fmin= 100, num_bands= 24, norm= True)
    from madmom.audio.signal import Signal
    onset_strength= (sodf(Signal(data= wav_data, sample_rate= sr)))
    onset_strength= librosa.util.normalize(onset_strength)
    h_length= int(librosa.time_to_samples(1./50, sr=sr))

    onset_times= librosa.onset.onset_detect(onset_envelope= onset_strength,
                                      sr=sr,
                                      hop_length= h_length,
                                      units='time', pre_max= 5, post_max= 5, 
                                      pre_avg= 5, post_avg= 5)

    return onset_times


def generate_notes(onset_times, ep_frames):
    notes = []
    onset_num= 0
    cur_frame= []
    cur_pitch= []

    for time, pitch in ep_frames:

        if (onset_num+ 1) < len(onset_times) and time > (onset_times[onset_num+ 1]- 0.016):

            note= Note(frame= cur_frame, frame_pitch= cur_pitch, onset_time= onset_times[onset_num]
                , offset_time= onset_times[onset_num+ 1])
            notes.append(note)
            
            cur_frame= []
            cur_pitch= []
            onset_num= onset_num+ 1

        if time > (onset_times[onset_num]- 0.016):
            cur_frame.append(time)
            cur_pitch.append(pitch)

    if cur_frame != []:
        note= Note(frame= cur_frame, frame_pitch= cur_pitch, onset_time= onset_times[onset_num]
            , offset_time= cur_frame[-1])
        notes.append(note)
        
    return notes

def get_note_level_pitch(notes):
    for note in notes:
        voiced_note= 0
        total= 0
        for i in range(len(note.frame_pitch)):
            if note.frame_pitch[i] > 0:
                voiced_note= voiced_note+ 1
                total= total+ note.frame_pitch[i]

        if voiced_note == 0:
            note.pitch= 0
        else:
            note.pitch= round( total / float(voiced_note) )

    return notes

def get_offset(notes):

    for note in notes:
        if note.pitch != 0:
            offset= 0
            for i in range(len(note.frame_pitch)):
                if note.frame_pitch[i] > 0:
                    offset= i
            
            if offset > 2:
                note.offset_time= note.frame[offset]

    return notes

def notes2txt(notes, filename):
    with open(filename, 'w') as file:
        for note in notes:
            if note.pitch != 0:
                file.write("%f %f %d\n" %(note.onset_time, note.offset_time, note.pitch))
    return

def notes2list(notes):
    res = []
    for note in notes:
        if note.pitch != 0:
            res.append([note.onset_time, note.offset_time, note.pitch])
    return res

def main(wav_path, ep_path):
    start_time = time.time()
    
    ep_frames = json.load(open(ep_path))

    onset_times = get_onset(wav_path)
    notes = generate_notes(onset_times, ep_frames)
    notes = get_note_level_pitch(notes)
    notes = get_offset(notes)

    print(wav_path, ep_path, time.time() - start_time)

    return notes2list(notes)

def file_check():
    vis = [0 for _ in range(1501)]
    for file in os.listdir('./AIcup_testlink'):
        if file.endswith('wav'):
            vis[int(file.split('.')[0])] = 1
    for i in range(1, len(vis)):
        if vis[i] == 0:
            print(i, end=' ')
    print()

if __name__ == '__main__':
    file_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="./log/example_out.json")
    args = parser.parse_args()
    dic = {}
    miss_list = []
    for root, dirs, files in os.walk('./AIcup_testset_ok'):
        for file_name in files:
            if file_name.endswith('_vocal.json'):
                file_id = int(file_name.split('_')[0])
                ep_path = os.path.join(root, file_name)
                wav_path = os.path.join('./AIcup_testlink', f'{file_id}.wav')
                if os.path.exists(wav_path):
                    dic[file_id] = main(wav_path=wav_path, ep_path=ep_path)
                else:
                    dic[file_id] = []
                    miss_list.append(file_id)
                    print(f'{file_id}.wav not exist!')
    with open(args.pred_file, "w") as f:
        json.dump(dic, f, sort_keys=True)
    print(sorted(miss_list))