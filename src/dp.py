import os
import sys
import json
import numpy as np

if __name__ == '__main__':
    data_path = sys.argv[1]
    num_song = int(sys.argv[2])
    output_path = sys.argv[3]

    predict = {}
    error_bound = 1200

    for idx in range(1, num_song + 1):
        json_path = os.path.join(data_path, f'{idx}', f'{idx}_feature.json')

        with open(json_path, 'r') as json_file:
            feature = json.loads(json_file.read())
        pitch = np.round(np.array(feature['vocal_pitch']))

        num_note = 1000
        num_pitch = pitch.shape[0]
        
        dp_table = np.zeros((num_note, num_pitch))
        backtrack = [[0 for _ in range(num_pitch)] for _ in range(num_note)]
        first_pitch = [0 for _ in range(num_note)]

        for i in range(1, num_note):
            dp_table[i][0] = np.inf
            
        for i in range(1, num_pitch):
            median = np.median(pitch[:i + 1])
            dp_table[0][i] = np.sum(np.abs(pitch[:i + 1] - median))

        for i in range(1, num_pitch):
            for j in range(1, num_note):
                if first_pitch[j] == 0:
                    dp_table[j][i] = dp_table[j - 1][i - 1]
                    backtrack[j][i] = j - 1
                    first_pitch[j] = i
                else:
                    median = np.median(pitch[first_pitch[j] : i + 1])
                    dp_table[j][i] = min(dp_table[j - 1][i - 1], dp_table[j - 1][first_pitch[j] - 1] + np.sum(np.abs(pitch[first_pitch[j] : i + 1] - median)))
                    
                    if dp_table[j][i] == dp_table[j - 1][i - 1]:
                        backtrack[j][i] = j - 1
                        first_pitch[j] = i
                    else:
                        backtrack[j][i] = j

        note_pos = 0
        loss = np.inf
        for i in range(num_note):
            if dp_table[i][num_pitch - 1] < error_bound:
                note_pos = i
                break
        
        music = []
        pitch_pos = num_pitch - 1

        while note_pos != 0:
            # print(note_pos, pitch_pos)
            pos = pitch_pos - 1
            while pos >= 0 and backtrack[note_pos][pos] == note_pos:
                pos -= 1
            
            median = np.round(np.median(pitch[pos + 1 : pitch_pos + 1]))
            if median != 0:
                onset = 0.016 + 0.032 * (pos + 1)
                offset = 0.016 + 0.032 * pitch_pos
                music = [[onset, offset, median]] + music
            
            pitch_pos = pos
            note_pos -= 1

        print('number of music notes in {:4d}-th song: {}'.format(idx, len(music)))
        predict[f'{idx}'] = music

        for m in music:
            print(m)

    with open(output_path, 'w') as output:
        json.dump(predict, output)