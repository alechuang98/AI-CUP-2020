import json

def check_json(path):
    miss_list = []
    with open(path, "r") as f:
        data = json.load(f)
        vis = [0 for _ in range(1501)]
        for key in data:
            vis[int(key)] = 1
        print(f'total {len(data)} of results')
        for i in range(1, len(vis)):
            if vis[i] == 0:
                miss_list.append(i)
        print(miss_list)
    return data, miss_list

def add_empty_list(path):
    data, miss_list = check_json(path)
    with open(path, "w") as f:
        for i in miss_list:
            data[i] = []
        json.dump(data, f)
