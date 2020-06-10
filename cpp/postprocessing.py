import json
import os
import numpy as np
import argparse

def main(data_path, tmp_path, output, delta=0):
    dic = {}
    for idx in range(1, 1501):
        tmp_file_path = os.path.join(tmp_path, f'{idx}.txt')
        res = []
        with open(tmp_file_path, "r") as f:
            for line in f:
                line = line.strip('\n').split(' ')
                line[0] = float(line[0]) + delta
                line[1] = float(line[1]) + delta
                line[2] = float(line[2])
                res.append(line)
        dic[idx] = res
    with open(output, "w") as f:
        json.dump(dic, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./AIcup_testset_ok')
    parser.add_argument("--tmp_path", default='./tmp')
    parser.add_argument("--output", default='./log/dp.json')
    parser.add_argument("--delta", type=float, default=0)
    args = parser.parse_args()
    main(args.data_path, args.tmp_path, args.output)