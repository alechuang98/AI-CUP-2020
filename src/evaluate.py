import json
import mir_eval
import os
import argparse

def parse_to_json(path_in='./MIR-ST500', path_out='./log/groundtruth.json'):
    dic = {}
    for root, dirs, files in os.walk(path_in):
        for file_name in files:
            if file_name.endswith('groundtruth.txt'):
                file_id = int(file_name.split('_')[0])
                with open(os.path.join(root, file_name), "r") as f:
                    res = []
                    for line in f:
                        res.append(line.strip('\n').split(' '))
                    dic[file_id] = res
    with open(path_out, "w") as f:
        json.dump(dic, f, sort_keys=True)

def main():
    parser = argparse.ArgumentParser()
    parse_to_json()

if __name__ == '__main__':
    main()
