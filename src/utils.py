import json

def check_json(path):
    with open(path) as f:
        data = json.load(f)
        vis = [0 for _ in range(1501)]
        for key in data:
            vis[int(key)] = 1
        print(f'total {len(data)} of results')
        miss_list = []
        for i in range(1, len(vis)):
            if vis[i] == 0:
                miss_list.append(i)
        print(miss_list)
