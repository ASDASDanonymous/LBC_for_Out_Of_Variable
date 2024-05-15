def load_jsonl(did, feature_name):
    jsonl_files = {}
    if feature_name:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'.../data/{did}_{mode}_feature_names.jsonl'
    else:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'.../data/{did}_{mode}.jsonl'
    return jsonl_files

data_id = 50_100
use_feature_names = True
jsonl_files = load_jsonl(data_id, use_feature_names)

if __name__ == '__main__':
    print(jsonl_files)