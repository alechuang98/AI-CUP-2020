import json
import mir_eval
import os
import argparse
import numpy as np

def parse_to_json(path_in='./MIR-ST500', path_out='./log/groundtruth.json'):
    dic = {}
    for root, dirs, files in os.walk(path_in):
        for file_name in files:
            if file_name.endswith('groundtruth.txt'):
                file_id = int(file_name.split('_')[0])
                with open(os.path.join(root, file_name), "r") as f:
                    res = []
                    for line in f:
                        res.append([float(x) for x in line.strip('\n').split(' ')])
                    dic[file_id] = res
    with open(path_out, "w") as f:
        json.dump(dic, f, sort_keys=True)

class Evaluator():
    def __init__(self, train_data_path='./MIR-ST500', groundtruth_path='./log/groundtruth.json'):
        if not os.path.exists(groundtruth_path):
            parse_to_json(train_data_path, groundtruth_path)
        self.ref = {}
        with open(groundtruth_path) as f:
            data = json.load(f)
            for key, val in data.items():
                self.ref[key] = np.array(val)

    def evaluate(self, pred):
        """
        pred.shape == (n, len_of_frame, 1, 3)
        """
        pred = pred.squeeze(pred)
        score = 0
        for key in sorted(self.ref, key=lambda i: int(i[0])):
            scores = mir_eval.transcription.evaluate(self.ref[key][:, 0:2], self.ref[key][:, 2], pred[:, 0:2], pred[:, 2])
            score += .2 * scores['Onset_F-measure'] + .6 * scores['F-measure_no_offset'] + .2 * scores['F-measure']
        return score / len(self.ref)

    def evaluate_file(self, pred_path):
        score = 0
        with open(pred_path) as f:
            data = json.load(f)
            for key, val in data.items():
                val = np.array(val)
                scores = mir_eval.transcription.evaluate(self.ref[key][:, 0:2], self.ref[key][:, 2], val[:, 0:2], val[:, 2])
                score += .2 * scores['Onset_F-measure'] + .6 * scores['F-measure_no_offset'] + .2 * scores['F-measure']
            return score / len(self.ref)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="./log/train.json")
    args = parser.parse_args()
    evaluator = Evaluator()
    print(evaluator.evaluate_file(args.pred_file))

if __name__ == '__main__':
    main()
