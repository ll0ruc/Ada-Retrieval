
### import extranal packages here
import logging
import setproctitle
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import time
import copy 
import cProfile, pstats
import io
import random
import math
import sys
from accelerate import Accelerator
from accelerate.utils import broadcast
### import modules defined in this project
from src.utils import argument_parser, logger, general
from src.data import seqrecdataset, adduserhistory, addnegsamples
from src.facility.trainer import Trainer
## https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def gen_exp_name(config):
    exp_name = config['model']
    
    if 'exp_name' in config:
        exp_name += '-' + config['exp_name']
    return exp_name

def fix_seed(config):
    if 'seed' in config:
        _seed = config['seed']
    else:
        _seed = 2022
    general.init_seed(_seed)

r'''
    If existing user2history is None, load user history from file_name.
    Otherwise returns user2history directly.
    Returns:
        user2history: An N-length ndarray (N is user count) of ndarray (of history items). 
                    E.g., User2History[user_id] is an ndarray of item history for user_id. 
'''
def get_user_history(user2history, config, default_name):
    logger = logging.getLogger(config['exp_name'])
    file_path = config['dataset_path']
    if user2history is None:
        _user_history_filename = default_name
        _user_history_data_format = config['train_file_format']
        if 'user_history_filename' in config:
            _user_history_filename = config['user_history_filename'] 
            _user_history_data_format = config.get('user_history_file_format', _user_history_data_format)
        logger.info("Loading user history from {0} ...".format(_user_history_filename))
        user2history = general.load_user_history(file_path, _user_history_filename, config['n_users'], _user_history_data_format)
        logger.info("Done. {0} of users have history.".format(len(user2history)))
    return user2history

 
def get_data_loader(config, task, add_history_trans, MyDataSet, file_path, file_name, user2history=None, return_graph=False):
    config = copy.deepcopy(config)
    config['data_loader_task'] = task
    config['data_format'] = config['{0}_file_format'.format(task)] 
    config['eval_protocol'] = config.get('{0}_protocol'.format(task), None)
    num_workers = config.get('num_workers_{0}'.format(task), config['num_workers'])
    transform = None 
    if config['eval_protocol'] == 'one_vs_all':
        config['n_sample_neg_{0}'.format(task)] = -1

    if config.get('n_sample_neg_{0}'.format(task), -1) > 0:
        transform = addnegsamples.AddNegSamples(
                config['n_users'], config['n_items'], config['n_sample_neg_{0}'.format(task)],
                user2history=user2history)
    else:
        transform = None

    if '{0}_batch_size'.format(task) in config:
        config['batch_size'] = config['{0}_batch_size'.format(task)]

    dataset = MyDataSet(config, path=file_path, filename=file_name, transform=transform)  

    if add_history_trans is not None:
        dataset.add_user_history_transform(add_history_trans)

    if return_graph:
        graph = dataset.get_graph()
        return graph

    collate_fn = None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=bool(config['shuffle_train']) if task == 'train' else False,
        pin_memory=config['pin_memory'],
        num_workers=num_workers,
        persistent_workers=config['persistent_workers'],
        worker_init_fn=set_worker_sharing_strategy if config['num_workers'] >= 1 else None,
        collate_fn=collate_fn
    )
        
    return data_loader

def need_user_history(config):
    r"""
        On what circumstances we need the user history:
            (1) In dynamic negative sampling, we need user history to filter out false negative;
            (2) and (3): if valid or test protocol is full candidate evaluation, we need user history to skip interacted items;
    """
    if config.get('n_sample_neg_train', 0) > 0 \
        or config.get('test_protocol', None) == 'one_vs_all' \
        or config.get('valid_protocol', None) == 'one_vs_all':
        return True
    return False


def main(config, accelerator):
    ## constants: 
    DATA_TRAIN_NAME, DATA_VALID_NAME, DATA_TEST_NAME = config.get('data_train_name', 'train'), config.get('data_valid_name', 'valid'), config.get('data_test_name', 'test')

    ## variables determined by context
    save_model = True
    ### end of declaration
    logger = logging.getLogger(config['exp_name'])
    file_path = config['dataset_path']
    output_path = config['output_path']   
    os.makedirs(output_path, exist_ok=True)
    ## data
    MyDataSet = seqrecdataset.SeqRecDataset
    user2history = None
    user2history = get_user_history(user2history, config, DATA_TRAIN_NAME)
    add_history_trans =  adduserhistory.AddUserHistory(user2history, config['seq_last'])
    if need_user_history(config):
        user2history = get_user_history(user2history, config, DATA_TRAIN_NAME)

    model_name = config['model']
    model = general.get_class_instance(model_name, 'src/model')(config)
    ## prepare train data loader
    train_data = get_data_loader(
        config, 'train', add_history_trans, MyDataSet, file_path, DATA_TRAIN_NAME,
        user2history=user2history, return_graph=not model.__optimized_by_SGD__
        )
    ## prepare valid data loader
    if model.__optimized_by_SGD__:
        valid_data = get_data_loader(
            config, 'valid', add_history_trans, MyDataSet, file_path, DATA_VALID_NAME,
            user2history=user2history
            )
    else:
        valid_data = None

    logger.info(model)

    trainer = Trainer(config, model, accelerator)

    if user2history is not None:
        trainer.set_user_history(user2history)

    if valid_data:
        trainer.reset_evaluator()
    try:
        trainer.fit(
            train_data, valid_data, save_model=save_model, verbose=config['verbose'],
            load_best_model=config['load_best_model'], model_file=config['model_file'] if config['load_best_model'] else None
        )
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt: stopping the training and start evaluating on the test set.')

    ## prepare test data loader
    test_data = get_data_loader(
        config, 'test', add_history_trans, MyDataSet, file_path, DATA_TEST_NAME, 
        user2history=user2history)

    trainer.reset_evaluator()
    test_data = trainer.accelerator.prepare(test_data)
    test_result = trainer.evaluate(test_data, load_best_model=save_model, verbose=config['verbose'])

    if accelerator.is_local_main_process:
        logger.info('best valid ' + f': {trainer.best_valid_result}')
        logger.info('test result' + f': {test_result}')

        result_file = os.path.join(output_path, 'result_{0}.{1}.{2}.tsv'.format(exp_name, logger_time_str, logger_rand))
        logger.info('Saving test result to {0} ...'.format(result_file))
        fp = open(result_file, 'w')
        for metirc, result in test_result.items():
            fp.write(str(metirc)+'\t'+str(result)+'\n')
        fp.close()

def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).sort_stats('cumtime').print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)

if __name__ == '__main__': 
    job_start_time = time.time()
    config = argument_parser.parse_arguments()

    if config['gpu_id']>=0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    accelerator = Accelerator()
    config['device'] = accelerator.device
    exp_name = gen_exp_name(config)
    config['task'] = config.get('task', 'train')
    setproctitle.setproctitle("AdaRetrieval-{0}-{1}".format(config['task'], exp_name))
    config['exp_name'] = exp_name
    logger_dir = 'output' if 'output_path' not in config else config['output_path']
    logger_tensor = torch.tensor([int(time.time()), random.randint(0,100)]).to(config['device'])
    logger_tensor = broadcast(logger_tensor, 0)
    logger_time_str = datetime.fromtimestamp(logger_tensor[0].item()).strftime("%Y-%m-%d_%H:%M:%S").replace(':', '')
    logger_rand = logger_tensor[1].item()
    config['logger_time_str'] = logger_time_str
    config['logger_rand'] = logger_rand
    mylog = logger.Logger(logger_dir, exp_name, logger_time_str, logger_rand, is_main_process=accelerator.is_local_main_process)
    if config['is_adaretrieval'] == 1:
        config['reco_batch_size'] = math.ceil(config['reco_total_size'] / config['reco_batch_cnt'])
    mylog.log(mylog.INFO, 'config='+str(config))
    fix_seed(config)
    pr = cProfile.Profile()
    pr.enable()    
    main(config, accelerator)
    pr.disable()
    
    if accelerator.is_local_main_process:
        profile_result = prof_to_csv(pr)
        profile_log_filename = mylog.filename.replace('.txt', '.prof')
        with open(profile_log_filename, 'w') as wt:
            wt.write(profile_result)
        ##=================== ending the program ======================//
        job_end_time = time.time()
        mylog.log(mylog.INFO, 'Mission complete. Time elapsed: {0:.2f} minutes.'.format((job_end_time - job_start_time)/60))
        mylog.remove_handles() 
    

