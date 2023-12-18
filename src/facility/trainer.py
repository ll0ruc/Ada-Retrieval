
import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import logging
from src.facility.evaluation.onepos import OnePositiveEvaluator
from src.utils.general import dict2str

class Trainer(object):
    def __init__(self, config, model, accelerator):
        self.config = config
        self.exp_name = config['exp_name'] if 'exp_name' in config else __name__
        self.model = model
        self.accelerator = accelerator
        self.logger = logging.getLogger(self.exp_name)
        self.learning_rate = config.get('learning_rate', 0)
        self.epochs = config.get('epochs', 0)
        self.eval_step = min(1, self.epochs)
        self.early_stop = config.get('early_stop', 0)
        self.valid_metric_bigger = True
        self.test_batch_size = config.get('batch_size', 0)

        self.device = config.get('device', None)
        if 'checkpoint_dir' in config:
            self.checkpoint_dir = os.path.join(config['output_path'], config['checkpoint_dir'])
        else:
            self.checkpoint_dir = os.path.join(config['output_path'], 'checkpoint_{0}_{1}'.format(config['logger_time_str'], config['logger_rand']))
        os.makedirs(self.checkpoint_dir, exist_ok=True) 
        
        ## filename to save models
        saved_model_file = '{}-{}.pth'.format(config['model'], self.exp_name)
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config.get('weight_decay', 0)
        ## the key metric that determines the early stop
        if 'key_metric' in config:
            self.key_metric = config['key_metric']
        else:
            self.key_metric = 'group_auc'
        
        self.best_valid_result = None 
        self.best_valid_score = None 
        
        self.start_epoch = 0
        self.cur_step = 1
        if self.model.__optimized_by_SGD__:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            self.model = self.accelerator.prepare(self.model)
        
        if 'grad_clip_value' in config and config['grad_clip_value'] > 0:
            self.grad_clip_value = config['grad_clip_value']
        else:
            self.grad_clip_value = None

        self.objective_controller = None

        if self.accelerator.is_local_main_process:
            if config['use_tensorboard'] > 0:
                tb_log_dir = self.checkpoint_dir
                self.tb_logger = SummaryWriter(log_dir=tb_log_dir)
                self.logger.info(f"tensorboard log file saved in {tb_log_dir}")
            else:
                self.tb_logger = None

    def set_user_history(self, user_history):
        self.user_history = user_history

    def reset_evaluator(self):
        config = self.config
        self.evaluator = OnePositiveEvaluator(
            config['metrics'],
            config,
            self.accelerator
        )

    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            self.logger.error('Training loss is nan')
            return True
        return False


    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = 4
        train_loss_output = ('epoch %d training' + ' [' + 'time' +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = ('train_loss%d' + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += 'train loss' + ': ' + des % losses
        return train_loss_output + ']'

    def _get_key_metric(self, results, key_metric=None):
        if key_metric is None:
            key_metric = self.key_metric
        return results[key_metric]
    
    def early_stopping(value, best, cur_step, max_step=4, bigger=True):
        r""" validation-based early stopping

        Args:
            value (float): current result
            best (float): best result. When no history (at the first epoch), this variable is None
            cur_step (int): the number of consecutive steps that did not exceed the best result
            max_step (int): threshold steps for stopping
            bigger (bool, optional): whether the bigger the better

        Returns:
            A tuple of:
            - float,
                best result after this step
            - int,
                the number of consecutive steps that did not exceed the best result after this step
            - bool,
                whether to stop
            - bool,
                whether to update
        """
        stop_flag = False
        update_flag = False
        if max_step > 0:
            if bigger:
                if best is None or value > best:
                    cur_step = 0
                    best = value
                    update_flag = True
                else:
                    cur_step += 1
                    if cur_step > max_step:
                        stop_flag = True
            else:
                if best is None or value < best:
                    cur_step = 0
                    best = value
                    update_flag = True
                else:
                    cur_step += 1
                    if cur_step >= max_step:
                        stop_flag = True
        else:   # disable early stop
            stop_flag = False
            update_flag = True
        return best, cur_step, stop_flag, update_flag

    def fit(
            self, train_data, valid_data=None,
            save_model=True, load_best_model=False, model_file=None, verbose=2
        ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it is None, the utils.early_stopping is invalid.
            save_model (bool, optional): whether to save the model parameters, default: True
            load_best_model (bool):
            model_file (string): 
            verbose (int): Control wether to show the progress of training epoch and evaluate epoch. 
                                 0: show nothing;  1: show basic progress message, but no tqdm progress bar; 2: show everything

 
        """
        logger = self.logger 
        
        if load_best_model:
            self.load_model(model_file)  

        train_data, valid_data = self.accelerator.prepare(train_data, valid_data)
        for epoch_idx in range(self.start_epoch, self.epochs):
            if (epoch_idx + 1) % self.eval_step == 0:  ## do evaluation per every eval_step epochs
                if epoch_idx == 0:
                   logger.debug('>> Valid before training...')
                valid_start_time = time.time()

                valid_result = self.evaluate(valid_data, load_best_model=False, verbose=verbose)
                valid_score = self._get_key_metric(valid_result, self.config['key_metric'])

                self.best_valid_score, self.cur_step, stop_flag, update_flag = Trainer.early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.early_stop, bigger=self.valid_metric_bigger
                )
                valid_end_time = time.time()
                valid_score_output = ("epoch %d evaluating" + " [time" + ": %.2fs, " + "%s: %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, self.key_metric, valid_score)
                valid_result_output = 'complete scores on valid set: \n' + dict2str(valid_result)

                logger.info(valid_score_output)
                logger.info(valid_result_output)
                if self.accelerator.is_local_main_process:
                    if self.config['use_tensorboard']:
                        for k, v in valid_result.items():
                            self.tb_logger.add_scalar(f'valid/{k}', v, epoch_idx)

                if update_flag:
                    if save_model:
                        self.accelerator.wait_for_everyone()
                        self.save_model(self.saved_model_file, self.optimizer, epoch_idx, self.cur_step, valid_result, self.config)
                    self.best_valid_result = valid_result
                else:
                    logger.info("No better score in the epoch. Patience: {0} / {1}".format(self.cur_step, self.early_stop))

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    logger.info(stop_output)
                    break

            # train
            # self.accelerator.print("\n>> epoch {}".format(epoch_idx + 1))
            logger.info(f'\n>> epoch {epoch_idx + 1}')
            training_start_time = time.time()

            total_loss = None
            iter_data = (
                tqdm(
                    enumerate(train_data),
                    total=len(train_data),
                    desc="Train",
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process
                ) if verbose == 2 else enumerate(train_data)
            )

            for batch_idx, inter_data in iter_data: 
                samples = {k:inter_data[v] for k,v in train_data.dataset.return_key_2_index.items()}
                self.model.train()
                global_step = len(iter_data) * (epoch_idx - self.start_epoch) + batch_idx

                loss, _, _, _ = self.model(**samples)
                
                self.optimizer.zero_grad()
               
                loss_is_nan = self._check_nan(loss)
                if not loss_is_nan:
                    self.accelerator.backward(loss)
                    if self.grad_clip_value is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    self.optimizer.step()
                else:
                    logger.info(f'Loss is nan. Do not update model and skip this step. Global step: {global_step}')
                    continue

                loss = self.accelerator.gather_for_metrics(loss.detach()).mean()
                total_loss = loss.item() if total_loss is None else total_loss + loss.item()
                if self.config['use_tensorboard'] and self.accelerator.is_local_main_process:
                    self.tb_logger.add_scalar('train/loss', loss.item(), global_step)

            training_end_time = time.time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx+1, training_start_time, training_end_time, total_loss)

            logger.info(train_loss_output)
    
    def load_model(self, filename=None):
        if filename:
            ## if the model file is specified, load the target mdoel
            checkpoint_file = filename
        else:
            ## load the best model trained in this experiment
            checkpoint_file = self.saved_model_file
        checkpoint = torch.load(checkpoint_file)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.logger.info('Loading model from {0}. The best epoch was {1}'.format(checkpoint_file, checkpoint['cur_epoch']))

    def save_model(self, filename, optimizer, cur_epoch=-1, cur_step=-1, best_valid_score=None, config=None):
        state = {
            'config': config,
            'cur_epoch': cur_epoch,
            'cur_step': cur_step,
            'best_valid_score': best_valid_score,
            'state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer': optimizer.optimizer.state_dict() if optimizer is not None else None,
        }
        success = False
        
        for _ in range(5):
            try:
                # torch.save(state, filename)
                self.accelerator.save(state, filename)
                self.logger.info('Saving best model at epoch {0} to {1}'.format(cur_epoch, filename))
                success = True
            except IOError:
                pass
            if success:
                break
        if not success:
            self.logger.error('Failed to save best model at epoch {0} to {1}'.format(cur_epoch, filename))

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, verbose=0):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            verbose (int): Control whether to show the progress of training epoch and evaluate epoch. 
                                 0: show nothing;  1: show basic progress message, but no tqdm progress bar; 2: show everything

        Returns:
            dict or numpy.ndarray: 
                if predict_only=False: eval result, key is the eval metric and value is the corresponding metric value;
                elif predict_only=True: a numpy array of predicted scores for instances.
        """ 
        if not eval_data:
            return
        
        eval_protocol = eval_data.dataset.config['eval_protocol'] 
                
        logger = self.logger
        logger.info(eval_protocol)
        
        if load_best_model:
            self.load_model(model_file)

        result = self.evaluator.evaluate_with_full_items(
            eval_data, self.model, self.user_history, verbose=verbose,
        )

        return result
