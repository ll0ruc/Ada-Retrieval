import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import yaml


r''' 
    Process data file: the last item is test set, the second last item is valid set, the rest are train set.
'''


def datasplit(infile, outdir):
    os.makedirs(outdir, exist_ok=True)
    train_file = os.path.join(outdir, 'train.txt')
    valid_file = os.path.join(outdir, 'valid.txt')
    test_file = os.path.join(outdir, 'test.txt')
    user_history_file = os.path.join(outdir, 'user_history.txt')

    wt_train = open(train_file, 'w')
    wt_valid = open(valid_file, 'w')
    wt_test = open(test_file, 'w')
    wt_user_history = open(user_history_file, 'w')
    lengths = []
    with open(infile, 'r') as rd:
        lines = rd.readlines()
        for line in lines:
            words = line.strip().split(' ')
            userid, items = words[0], words[1:]
            item_set = set()
            dedup_items = []
            for item in items:
                if item not in item_set:
                    item_set.add(item)
                    dedup_items.append(item)
            items = dedup_items
            if len(items) < 3:
                print(userid)
                continue
            lengths.append(len(items))
            wt_train.write(userid + ' ' + ','.join(items[:-2]) + '\n')
            wt_valid.write(userid + ' ' + items[-2] + '\n')
            wt_test.write(userid + ' ' + items[-1] + '\n')
            wt_user_history.write(userid + ' ' + ','.join(items) + '\n')

    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()
    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')


def convert_to_pandas_pkl_file(
        infile, outpath, outfilename,
        names=None, data_format=None,
        sep=' '
):
    # dtypes = {'user_id':int, 'item_id':int}

    data = pd.read_csv(
        infile, sep=sep, header=None,
        names=names,
        engine='python'
    )

    print('data shape of {0} is {1}'.format(os.path.basename(infile), data.shape))
    print('data dtypes is {0}'.format(data.dtypes))

    ### additional data transformation in dataframe
    if data_format == "user-item_seq":
        data.item_seq = data.item_seq.apply(lambda t: np.array([int(a) for a in t.split(',')]))

    os.makedirs(outpath, exist_ok=True)
    max_user_id = data['user_id'].max()
    if data_format == "user-item":
        max_item_id = data['item_id'].max()
    else:
        max_item_id = data['item_seq'].apply(lambda t: np.max(t)).max()

    ## int64 is not JSON serializable
    max_user_id = int(max_user_id)
    max_item_id = int(max_item_id)

    n_lines = int(len(data))

    info = defaultdict(
        int,
        {
            'n_users': max_user_id + 1,
            'n_items': max_item_id + 1,
            'n_lines_{0}'.format(outfilename): n_lines
        }
    )
    info_file = os.path.join(outpath, 'data.info')
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as rd:
                pre_info = defaultdict(int, json.load(rd))
        except:
            pre_info = defaultdict(int)
        for key in info.keys():
            if key not in pre_info:
                pre_info[key] = info[key]
            else:
                if key.startswith('n_'):
                    pre_info[key] = max(info[key], pre_info[key])
                else:
                    if pre_info[key] != info[key]:
                        raise ValueError('key duplicated: {0}'.format(key))
        info = pre_info
    info['{0}_file_format'.format(outfilename)] = data_format

    with open(info_file, 'w') as wt:
        json.dump(info, wt)

    data = data.reset_index(drop=True)
    data.to_feather(os.path.join(outpath, outfilename + '.ftr'))
    print('In saving:')
    print(data.head(5))
    print('data.shape={0}\n'.format(data.shape))
    return info



def process_transaction_dataset(dataset_name, raw_datapath, dataset_outpathroot, example_yaml_file):
    sep = " "
    np.set_printoptions(linewidth=np.Inf)

    for data_name in ['train', 'valid', 'test', 'user_history']:
        if data_name in ['train', 'user_history']:
            names = ['user_id', 'item_seq']
            data_format = 'user-item_seq'
        else:
            names = ['user_id', 'item_id']
            data_format = 'user-item'
        info = convert_to_pandas_pkl_file(
            os.path.join(raw_datapath, '{0}.txt'.format(data_name)),
            dataset_outpathroot,
            data_name,
            names=names,
            data_format=data_format,
            sep=sep
        )
        os.remove(os.path.join(raw_datapath, '{0}.txt'.format(data_name)))

    with open(example_yaml_file, 'r') as rd:
        data_config = yaml.safe_load(rd)
    data_config['n_users'] = info['n_users']
    data_config['n_items'] = info['n_items']
    for k, v in info.items():
        if '_file_format' in k:
            data_config[k] = v

    dataset_yaml_file = os.path.join(
        os.path.dirname(example_yaml_file),
        dataset_name + '.yaml'
    )
    with open(dataset_yaml_file, 'w') as wt:
        yaml.dump(data_config, wt, default_flow_style=False)



def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--dataset_name', default="Beauty", type=str)
    arguments.add_argument('--infile', default="../data/Beauty/Beauty.txt", type=str)
    arguments.add_argument('--outdir', default="../data/Beauty/", type=str)
    arguments.add_argument('--example_yaml', default="../src/config/dataset/example.yaml", type=str)
    args = arguments.parse_args()
    datasplit(args.infile, args.outdir)
    raw_datapath =args.outdir
    dataset_name = args.dataset_name
    dataset_outpathroot = args.outdir
    example_yaml_file = args.example_yaml
    process_transaction_dataset(dataset_name, raw_datapath, dataset_outpathroot, example_yaml_file)


if __name__ == '__main__':
    main()