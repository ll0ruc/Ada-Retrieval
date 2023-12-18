import argparse
import os

from src.utils import file_io

def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
    ## define general arguments
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--config_file", type=str, help='yaml file to store the default configuration. config_file and config_dir are conflicting. Only need to specify one.')  
    parser.add_argument("--config_dir", type=str, help='yaml folder to store the default configuration. config_file and config_dir are conflicting. Only need to specify one.')  
    parser.add_argument("--seed", type=int, help='the random seed to be fixed for reproducing experiments.') 
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str, default='train')
    parser.add_argument("--use_tensorboard", type=int, default=0, choices=[0,1])

    ### training specific arguments
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_workers_train", type=int)
    parser.add_argument("--num_workers_valid", type=int)
    parser.add_argument("--num_workers_test", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--grad_clip_value", type=float)
    parser.add_argument("--score_clip_value", type=float)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--metrics", type=str)
    parser.add_argument("--key_metric", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--shuffle_train", type=int)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--init_method", type=str)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--init_mean", type=float, default=0.0)
    
    ### data specific arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_path", type=str) #DATA_TRAIN_NAME
    parser.add_argument("--data_train_name", type=str)
    parser.add_argument("--data_valid_name", type=str)
    parser.add_argument("--data_test_name", type=str)
    parser.add_argument("--output_path", type=str)  
    parser.add_argument("--train_file_format", type=str) 
    parser.add_argument("--valid_file_format", type=str)
    parser.add_argument("--test_file_format", type=str)
    parser.add_argument("--user_history_file_format", type=str)
    parser.add_argument("--user_history_filename", type=str)

    parser.add_argument("--test_protocol", type=str)
    parser.add_argument("--valid_protocol", type=str)
    parser.add_argument("--n_sample_neg_train", type=int)

    parser.add_argument("--model_file", type=str)
    parser.add_argument("--load_best_model", type=int, default=0, help="In the case that we need to load a pretrained model and continue training")
    parser.add_argument("--seq_last", type=int, default=0, help="whether to only take the last occurrence of an item in the sequence as the target.")

    ## define model-sepcific related arguments here 
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--valid_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--dropout_prob", type=float)
    parser.add_argument("--hidden_dropout_prob", type=float) ## in SASRec
    parser.add_argument("--attn_dropout_prob", type=float) ## in SASRec   
    parser.add_argument("--embedding_size", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--has_item_bias", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--inner_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--layer_norm_eps", type=float)
    parser.add_argument("--hidden_act", type=str)
    parser.add_argument("--gru_hidden_size", type=int)
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature parameter for softmax type loss")
    parser.add_argument("--step", type=int) ## For SRGNN
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--block_num", type=int)
    parser.add_argument("--dilations", type=int)
    parser.add_argument("--checkpoint_dir", type=str)

    ## define ada-retrieval related arguments here
    parser.add_argument("--reco_total_size", type=int)
    parser.add_argument("--reco_batch_cnt", type=int)
    parser.add_argument("--is_adaretrieval", type=int)
    parser.add_argument("--lambda", type=float)

    (args, unknown) = parser.parse_known_args()
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and value not in ['none', 'None']:
            parsed_results[arg] = value
    
    return parsed_results


def parse_arguments():
    cmd_arg = parse_cmd_arguments()    
    if 'config_file' in cmd_arg:
        config = file_io.load_yaml(cmd_arg['config_file']) 
    elif 'config_dir' in cmd_arg:
        ## the priority is: dataset > model > base.yaml
        config = file_io.load_yaml(os.path.join(cmd_arg['config_dir'], 'base.yaml'))
        config.update(file_io.load_yaml(os.path.join(cmd_arg['config_dir'], 'model', cmd_arg['model']+'.yaml')))
        config.update(file_io.load_yaml(os.path.join(cmd_arg['config_dir'], 'dataset', cmd_arg['dataset']+'.yaml')))                 
    else:
        config = {}
    config.update(cmd_arg)
    return config
