import os
import time
import json

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from .model import CoQAModel, QuACModel
from .utils import prepare_datasets, sanitize_input, vectorize_input, vectorize_input_turn
# from .utils.process_utils import ExampleProcessor
from .utils import Timer, DummyLogger, AverageMeter
from .utils import constants as Constants


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """

    def __init__(self, config):
        config['dataset_name'] = config['dataset_name'].lower()
        if config['dataset_name'] == 'coqa':
            QAModel = CoQAModel
            # Evaluation Metrics:
            self._train_loss = AverageMeter()
            self._dev_loss = AverageMeter()
            self._train_metrics = {'f1': AverageMeter(),
                                'em': AverageMeter()}
            self._dev_metrics = {'f1': AverageMeter(),
                                'em': AverageMeter()}
            config['coqa_answer_class_num'] = Constants.CoQA_ANSWER_CLASS_NUM

        elif config['dataset_name'] in ('quac', 'doqa'):
            QAModel = QuACModel
            # Evaluation Metrics:
            self._train_loss = AverageMeter()
            self._dev_loss = AverageMeter()
            self._train_metrics = {'f1': AverageMeter(),
                                'heq': AverageMeter(),
                                'dheq': AverageMeter()}
            self._dev_metrics = {'f1': AverageMeter(),
                                'heq': AverageMeter(),
                                'dheq': AverageMeter()}

            config['quac_yesno_class_num'] = Constants.QuAC_YESNO_CLASS_NUM
            if config['dataset_name'] == 'quac':
                config['quac_followup_class_num'] = Constants.QuAC_FOLLOWUP_CLASS_NUM
            else:
                config['quac_followup_class_num'] = Constants.DoQA_FOLLOWUP_CLASS_NUM

        else:
            raise ValueError('Unknown dataset name: {}'.format(config['dataset_name']))

        # self.example_processor = ExampleProcessor()
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        # Prepare datasets
        datasets = prepare_datasets(config)
        train_set = datasets['train']
        dev_set = datasets['dev']
        test_set = datasets['test']
        self.test_set = test_set

        if train_set:
            self.train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                                           shuffle=config['shuffle'], collate_fn=lambda x: x, pin_memory=True)
            self._n_train_batches = len(train_set) // config['batch_size']
        else:
            self.train_loader = None

        if dev_set:
            self.dev_loader = DataLoader(dev_set, batch_size=config['batch_size'],
                                         shuffle=False, collate_fn=lambda x: x, pin_memory=True)
            self._n_dev_batches = len(dev_set) // config['batch_size']
        else:
            self.dev_loader = None

        if test_set:
            self.test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False,
                                          collate_fn=lambda x: x, pin_memory=True)
            self._n_test_batches = len(test_set) // config['batch_size']
            self._n_test_examples = len(test_set)
        else:
            self.test_loader = None

        # Load BERT featrues
        if config['use_bert']:
            from pytorch_pretrained_bert import BertTokenizer
            from pytorch_pretrained_bert.modeling import BertModel
            print('[ Using pretrained BERT features ]')
            self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_model'], do_lower_case=True)
            self.bert_model = BertModel.from_pretrained(config['bert_model']).to(self.device)
            config['bert_model'] = self.bert_model
            if not config.get('finetune_bert', None):
                print('[ Fix BERT layers ]')
                self.bert_model.eval()
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            else:
                print('[ Finetune BERT layers ]')
        else:
            self.bert_tokenizer = None
            self.bert_model = None

        # Initialize the QA model
        self._n_train_examples = 0
        self.model = QAModel(config, train_set)
        self.model.network = self.model.network.to(self.device)
        self.config = self.model.config
        self.is_test = False

    def train(self):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return

        self.is_test = False
        timer = Timer("Train")
        if self.config['pretrained']:
            self._epoch = self._best_epoch = self.model.saved_epoch
        else:
            self._epoch = self._best_epoch = 0

        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = self._dev_metrics[k].mean()
        self._reset_metrics()

        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1

            print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])
            train_epoch_time = timer.interval("Training Epoch {}".format(self._epoch))
            format_str = "Training Epoch {} -- Loss: {:0.4f}".format(self._epoch, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)

            print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'])
            timer.interval("Validation Epoch {}".format(self._epoch))
            format_str = "Validation Epoch {} -- Loss: {:0.4f}".format(self._epoch, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)

            early_stop_metric = self.config.get('early_stop_metric', 'f1')
            self.model.scheduler.step(self._dev_metrics[early_stop_metric].mean())
            if self._best_metrics[early_stop_metric] <= self._dev_metrics[early_stop_metric].mean():  # Can be one of loss, f1, or em.
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if self.config['save_params']:
                    self.model.save(self.dirname, self._epoch)
                    print('Saved model to {}'.format(self.dirname))
                format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                self.logger.write_to_file(format_str)
                print(format_str)

            self._reset_metrics()

        timer.finish()
        self.training_time = timer.total

        print("Finished Training: {}".format(self.dirname))
        print(self.summary())
        return self._best_metrics

    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        print('Restoring best model')
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)


        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")

        output = []
        if self.config['use_predicted_answers']:
            for paragraph in self.test_set:
                paragraph_output = self._run_one_paragraph(paragraph=paragraph)
                output.append(paragraph_output)
        else:
            output = self._run_epoch(self.test_loader, training=False, verbose=0,
                                    out_predictions=self.config['out_predictions'])

        if self.config['out_predictions']:
            if self.config['out_pred_in_folder']:
                if self.config['use_predicted_answers']:
                    output_file = os.path.join(self.dirname, Constants._PRED_PREDICTION_FILE)
                else:
                    output_file = os.path.join(self.dirname, Constants._GOLD_PREDICTION_FILE)
            else:
                if self.config['use_predicted_answers']:
                    output_file = Constants._PRED_PREDICTION_FILE
                else:
                    output_file = Constants._GOLD_PREDICTION_FILE
            with open(output_file, 'w') as outfile:
                if self.config['dataset_name'] == 'coqa':
                    json.dump(output, outfile, indent=4)
                else:
                    for pred in output:
                        outfile.write(json.dumps(pred) + '\n')

        timer.finish()
        print(self.self_report(self._n_test_batches, 'test'))
        print("Finished Testing: {}".format(self.dirname))
        self.logger.close()

        test_metrics = {}
        for k in self._dev_metrics:
            test_metrics[k] = self._dev_metrics[k].mean()
        return test_metrics

    def _run_one_paragraph(self, paragraph):
        output = []
        QA_history = []
        history=[]
        cid = paragraph['id']
        evidence = paragraph['evidence']
        raw_evidence = paragraph['raw_evidence']

        qid_list = []
        best_span_str_list = []
        yesno_list = []
        followup_list = []
        for turn, qa in enumerate(paragraph['turns']):
            example = {'id': cid,
                        'evidence': evidence,
                        'raw_evidence': raw_evidence}
            temp = []
            marker = []

            n_history = len(history) if self.config['n_history'] < 0 else min(
                self.config['n_history'], len(history))
            if n_history > 0:
                count = sum([not self.config['no_pre_question'], not self.config['no_pre_answer']]) * len(history[-n_history:])
                for q, a in history[-n_history:]:
                    if not self.config['no_pre_question']:
                        temp.extend(q)
                        marker.extend([count] * len(q))
                        count -= 1
                    if not self.config['no_pre_answer']:
                        temp.extend(a)
                        marker.extend([count] * len(a))
                        count -= 1
            original_question = qa['question']['word']
            temp.extend(original_question)
            marker.extend([0] * len(original_question))
            qa['question']['word'] = temp
            qa['question']['marker'] = marker


            example['turns'] = [qa]
            test_loader = DataLoader([example], batch_size=1, shuffle=False,
                                          collate_fn=lambda x: x, pin_memory=True)
            for step, input_batch in enumerate(test_loader):
                input_batch = sanitize_input(input_batch, self.config, self.model.word_dict,
                                            self.model.feature_dict, self.bert_tokenizer, training=False)
                x_batch = vectorize_input_turn(
                    input_batch, self.config, self.bert_model, QA_history, turn, device=self.device)
                if not x_batch:
                    continue  # When there are no target spans present in the batch

                res = self.model.predict(
                    x_batch, step, update=False, out_predictions=True)

                for idx, turn_ids in enumerate(input_batch['turn_ids']): # should be only one

                    for t_idx, t_id in enumerate(turn_ids):
                        qid_list.append(t_id)
                        best_span_str_list.append(res['predictions'][idx][t_idx])
                        yesno_list.append(res['yesnos'][idx][t_idx])
                        followup_list.append(res['followups'][idx][t_idx])
                        QA_history.append(res['token_spans'][idx][t_idx])
                        answer_tokens = res['token_lists'][idx][t_idx]
                        history.append((original_question, answer_tokens))

        output = {'qid': qid_list,
                'best_span_str': best_span_str_list,
                'yesno': yesno_list,
                'followup': followup_list}
        return output

    def _run_epoch(self, data_loader, training=True, verbose=10, out_predictions=False):
        start_time = time.time()
        if training:
            self.model.optimizer.zero_grad()
        output = []
        for step, input_batch in enumerate(data_loader):
            input_batch = sanitize_input(input_batch, self.config, self.model.word_dict,
                                         self.model.feature_dict, self.bert_tokenizer, training=training)
            x_batch = vectorize_input(input_batch, self.config, self.bert_model, training=training, device=self.device)
            if not x_batch:
                continue  # When there are no target spans present in the batch

            res = self.model.predict(x_batch, step, update=training, out_predictions=out_predictions)

            loss = res['loss']
            metrics = res['metrics']
            self._update_metrics(loss, metrics, res['total_qs'], res['total_dials'], training=training)

            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                mode = "train" if training else ("test" if self.is_test else "dev")
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('used_time: {:0.2f}s'.format(time.time() - start_time))

            if out_predictions:
                if self.config['dataset_name'] == 'coqa':
                    for idx, (id, turn_ids) in enumerate(zip(input_batch['id'], input_batch['turn_ids'])):
                        for t_idx, t_id in enumerate(turn_ids):
                            output.append({'id': id,
                                           'turn_id': t_id,
                                           'answer': res['predictions'][idx][t_idx]})
                else:
                    for idx, turn_ids in enumerate(input_batch['turn_ids']):
                        qid_list = []
                        best_span_str_list = []
                        yesno_list = []
                        followup_list = []
                        for t_idx, t_id in enumerate(turn_ids):
                            qid_list.append(t_id)
                            best_span_str_list.append(res['predictions'][idx][t_idx])
                            yesno_list.append(res['yesnos'][idx][t_idx])
                            followup_list.append(res['followups'][idx][t_idx])

                        output.append({'qid': qid_list,
                                       'best_span_str': best_span_str_list,
                                       'yesno': yesno_list,
                                       'followup': followup_list})
        return output

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.4f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.4f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                    self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.2f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.2f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, total_qs, total_dials, training=True):
        if training:
            self._train_loss.update(loss)
            for k in self._train_metrics:
                self._train_metrics[k].update(metrics[k] * 100, total_qs if k.lower() != 'dheq' else total_dials)
        else:
            self._dev_loss.update(loss)
            for k in self._dev_metrics:
                self._dev_metrics[k].update(metrics[k] * 100, total_qs if k.lower() != 'dheq' else total_dials)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True
