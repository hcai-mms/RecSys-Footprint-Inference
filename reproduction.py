import os
import sys
import argparse
import csv
import datetime
import random

import numpy as np
from codecarbon import EmissionsTracker

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from recbole.model.general_recommender import Pop, ItemKNN, BPR, DMF, FISM, NAIS, SLIMElastic
from recbole.model.general_recommender import NeuMF, MultiDAE, NGCF, DGCF, LightGCN, SGL
from recbole.model.knowledge_aware_recommender import CKE, CFKG, KGCN, KTUP, RippleNet

from recbole.data.dataloader.knowledge_dataloader import KGDataLoaderState
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model


VALID_MODELS = {
    'pop': 'Pop',
    'itemknn': 'ItemKNN',
    'bpr': 'BPR',
    'dmf': 'DMF',
    'fism': 'FISM',
    'nais': 'NAIS',
    'slim': 'SLIMElastic', 'slimelastic': 'SLIMElastic',
    'neumf': 'NeuMF',
    'multidae': 'MultiDAE',
    'ngcf': 'NGCF',
    'dgcf': 'DGCF',
    'lighgcn': 'LightGCN',
    'sgl': 'SGL',
    'cke': 'CKE',
    'cfkg': 'CFKG',
    'kgcn': 'KGCN',
    'ktup': 'KTUP',
    'ripplenet': 'RippleNet'
}

GENERAL_MODELS = ['Pop', 'ItemKNN', 'BPR', 'DMF', 'FISM', 'NAIS', 'SLIMElastic', 'NeuMF', 'MultiDAE',
                  'NGCF', 'DGCF', 'LightGCN', 'SGL']

KNOWLEDGE_MODELS = ['CKE', 'CFKG', 'KGCN', 'KTUP', 'RippleNet']

VALID_DATASETS = ['amazon', 'mind', 'movielens']

TRAINING_LOG_FIELDNAMES = ['counter', 'timestamp', 'duration', 'dataset', 'model', 'test_result', 'parameter_dict',
                           'model_path', 'energy_consumed', 'country_iso_code']

INFERENCE_LOG_FIELDNAMES = ['users', 'results', 'scores', 'kwh_consumed', 'duration']


def init_training_log_and_counter(logfile):
    """
    looks for running number of next experiment in specified log file
    if logfile does not exist it is created and counter set to zero
    :param logfile: str
    :return counter: number of next experiment
    """
    logfile = logfile + '.csv'
    # create file if it does not exist
    if not os.path.exists(logfile):
        with open(logfile, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=TRAINING_LOG_FIELDNAMES)
            writer.writeheader()
        counter = 0
    # otherwise identify next running number
    else:
        with open(logfile, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            last_row = None
            for row in reader:
                last_row = row
        if last_row is None:
            counter = 0
        else:
            counter = int(last_row['counter']) + 1
    return counter

def read_training_log(dataset):
    """
    reads training log and returns list of dictionary entries
    :param dataset: str
    :return log: list of dictionaries
    """
    logfile = dataset + '_training_log.csv'
    log = []
    with open(logfile, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for log_entry in reader:
            log.append(log_entry)
    return log

def write_training_log(logfile, counter, timestamp, duration, dataset, model, test_result, parameter_dict,
                       model_path, energy_consumed, country_iso_code):
    """
    appends log entry of training experiment to existing log file
    :param logfile: file to use for append
    :param TRAINING_LOG_FIELDNAMES: all log fields
    :return:
    """
    logfile = logfile + '.csv'
    with open(logfile, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=TRAINING_LOG_FIELDNAMES)
        writer.writerow({'counter': counter, 'timestamp': timestamp, 'duration': duration,
                         'dataset': dataset, 'model': model, 'test_result': test_result,
                         'parameter_dict': parameter_dict, 'model_path': model_path,
                         'energy_consumed': energy_consumed, 'country_iso_code': country_iso_code})


class TrainingExperiment:

    def __init__(self, dataset_to_use=None, model_to_use=None):
        """
        data for training experiment object
        :param dataset_to_use:
        :param model_to_use:
        """
        self.dataset = dataset_to_use
        self.model = model_to_use
        self.log_file = self.dataset + '_training_log'
        self.counter = init_training_log_and_counter(self.log_file)
        self.project_name = str(self.counter) + '_' + self.dataset + '_' + self.model
        self.parameter_dict = None
        self.timestamp = None
        self.duration = None
        self.model_file = None
        self.best_valid_score = None
        self.best_valid_result = None
        self.test_result = None
        self.energy_consumed = None
        self.country_iso_code = None

    def train_model(self, parameter_dict=None, metrics='all'):
        """
        model training execution
        :param parameter_dict: optional, to override preset parameter_dict
        :param metrics: 'all' (12 metrics) or 'one' (MRR only)
        :return:
        """

        # get all previously saved models
        dir_saved = os.listdir('./saved/' + self.dataset + '/')
        all_models_before = [k for k in dir_saved if k.startswith(self.model)]

        # if optional parameter_dict is passed, override following preset dictionaries
        if parameter_dict is not None:
            self.parameter_dict = parameter_dict

        # otherwise use the following preset parameter_dict depending on algorithm
        # can be used for algorithm-specific settings (like 'epochs': 1 for Pop)
        # setting 'stopping_step': 50 switches off early stopping on purpose
        elif self.model == 'Pop':
            self.parameter_dict = {
                'epochs': 1,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'ItemKNN':
            self.parameter_dict = {
                'epochs': 1,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'BPR':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'DMF':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'FISM':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'NAIS':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'SLIMElastic':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'NeuMF':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'MultiDAE':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'NGCF':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'DGCF':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'LightGCN':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'SGL':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'save_dataloaders': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'CKE':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'CFKG':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'KGCN':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'KTUP':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        elif self.model == 'RippleNet':
            self.parameter_dict = {
                # 'epochs': 1, # for testing
                'epochs': 20 if self.dataset == 'amazon' else 50,
                'stopping_step': 50,
                'checkpoint_dir': './saved/' + self.dataset + '/',
                'save_dataset': True,
                'metrics': ['MRR'] if metrics == 'one' else
                ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                 'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage']
            }

        else:
            raise ValueError('Not a valid model!')

        # RecBole configurations initialization
        config = Config(model=self.model, dataset=self.dataset, config_dict=self.parameter_dict)

        # init random seed and set for reproducibility data splits
        init_seed(config['seed'], config['reproducibility'])

        # logger initialization
        init_logger(config)
        logger = getLogger()
        # write config info into log
        logger.info(config)

        # create dataset and log its statistics
        dataset = create_dataset(config)
        logger.info(dataset)

        # prepare data splits (train/valid/test)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # tweak for knowledge-aware models to avoid error
        if self.model in KNOWLEDGE_MODELS:
            # patch only the training dataloader (which is knowledge-based)
            # do not attempt to patch valid_data and test_data, since they are evaluation loaders
            if hasattr(train_data, '_dataset'):
                object.__setattr__(train_data, 'dataset', train_data._dataset)
            # for training loader set its mode to combined state (RSKG)
            if hasattr(train_data, 'set_mode'):
                train_data.set_mode(KGDataLoaderState.RSKG)

            # dynamically get model class and instantiate it with the dataset
            model_class = get_model(config['model'])
            model = model_class(config, dataset).to(config['device'])

        # for general recommender models
        else:
            
            # dynamically get model class and instantiate it with the dataset
            model_class = get_model(config['model'])
            model = model_class(config, train_data.dataset).to(config['device'])

        logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(config, model)

        # timestamp
        self.timestamp = str(datetime.datetime.now())

        # execute model training with CodeCarbon tracker
        with EmissionsTracker(project_name=self.project_name,
                              output_file=self.dataset + '_training_emissions_log.csv') as tracker:
            self.best_valid_score, self.best_valid_result = trainer.fit(train_data=train_data,
                                                                        valid_data=valid_data)

        # collect CodeCarbon data
        self.duration = tracker.final_emissions_data.duration
        self.energy_consumed = tracker.final_emissions_data.energy_consumed
        self.country_iso_code = tracker.final_emissions_data.country_iso_code

        # model evaluation
        self.test_result = dict(trainer.evaluate(test_data))

        # collect data for log
        dir_saved = os.listdir('./saved/' + self.dataset + '/')
        all_models_after = [k for k in dir_saved if k.startswith(self.model)]
        s = set(all_models_before)
        new_model_file = [k for k in all_models_after if k not in s]
        self.model_file = './saved/' + self.dataset + '/' + new_model_file[0]
        # write log
        write_training_log(logfile=self.log_file, counter=self.counter, timestamp=self.timestamp,
                           duration=self.duration, dataset=self.dataset, model=self.model,
                           test_result=self.test_result, parameter_dict=self.parameter_dict,
                           model_path=self.model_file, energy_consumed=self.energy_consumed,
                           country_iso_code=self.country_iso_code)

        # end logger
        logger.handlers.clear()


class EvaluationExperiment:

    def __init__(self, dataset_to_use, model_file):
        """
        data for inference experiment
        :param dataset_to_use:
        :param model_file:
        """
        self.dataset = dataset_to_use
        self.model_file = None

        # read training log and extract model infos and path from number or name
        log = read_training_log(self.dataset)
        if isinstance(model_file, int) or model_file.isnumeric():
            for log_entry in log:
                if log_entry['counter'] == str(model_file):
                    self.counter = str(model_file)
                    self.model_file = log_entry['model_path']
                    self.model = log_entry['model']
                    self.parameter_dict = eval(log_entry['parameter_dict'])
        else:
            for log_entry in log:
                if model_file in log_entry['model_path'].lower():
                    self.model_file = log_entry['model_path']
                    self.counter = log_entry['counter']
                    self.model = log_entry['model']
                    self.parameter_dict = eval(log_entry['parameter_dict'])

        self.valid_result = None
        self.test_result = None

    def evaluate_model(self, metrics=None):
        """
        evaluates performance metrics
        :param metrics: optional list of metrics, otherwise set of 12 is computed
        """

        # error detection
        if self.model_file is None:
            print('\nexperiment number does not exist.')
            return
        if not os.path.exists(self.model_file):
            print('\nmodel file does not exist.')
            return

        # check for metrics list override, otherwise compute set of 12
        if metrics is not None:
            self.parameter_dict['metrics'] = metrics
        else:
            self.parameter_dict['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision',
                                              'GAUC', 'ItemCoverage', 'AveragePopularity',
                                              'GiniIndex', 'ShannonEntropy', 'TailPercentage']

        # set RecBole configuration for reproducibility
        config = Config(model=self.model, dataset=self.dataset, config_dict=self.parameter_dict)
        init_seed(config['seed'], config['reproducibility'])
        # create and prepare dataset
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # tweak for knowledge-aware models to avoid error
        if self.model in KNOWLEDGE_MODELS:
            # patch only the training dataloader (which is knowledge-based)
            # do not attempt to patch valid_data and test_data, since they are evaluation loaders
            if hasattr(train_data, '_dataset'):
                object.__setattr__(train_data, 'dataset', train_data._dataset)
            # for training loader set its mode to combined state (RSKG)
            if hasattr(train_data, 'set_mode'):
                train_data.set_mode(KGDataLoaderState.RSKG)

            # dynamically get model class and instantiate it with the dataset
            model_class = get_model(config['model'])
            model = model_class(config, dataset).to(config['device'])

        # for general recommender models
        else:

            # dynamically get model class and instantiate it with the dataset
            model_class = get_model(config['model'])
            model = model_class(config, train_data.dataset).to(config['device'])

        # trainer loading and initialization
        trainer = Trainer(config, model)
        trainer.eval_collector.data_collect(train_data)

        # model evaluation on validation and test sets
        self.valid_result = dict(trainer.evaluate(eval_data=valid_data, model_file=self.model_file, show_progress=True))
        self.test_result = dict(trainer.evaluate(eval_data=test_data, model_file=self.model_file, show_progress=True))


class InferenceExperiment:

    def __init__(self, dataset_to_use, model_file, topk=10):
        """
        data for inference experiment
        :param dataset_to_use:
        :param model_file:
        :param topk: set k for top-k items to compute
        """
        self.dataset = dataset_to_use
        self.model_file = None

        # read training log and extract model infos and path from number or name
        log = read_training_log(self.dataset)
        if isinstance(model_file, int) or model_file.isnumeric():
            for log_entry in log:
                if log_entry['counter'] == str(model_file):
                    self.counter = str(model_file)
                    self.model_file = log_entry['model_path']
        else:
            for log_entry in log:
                if model_file in log_entry['model_path'].lower():
                    self.model_file = log_entry['model_path']
                    self.counter = log_entry['counter']

        # initialize data
        self.topk = topk
        self.user_list = None
        self.result_list = None
        self.scores_list = None
        self.kwh_consumed = None
        self.kwh_list = []
        self.kwh_mean = None
        self.kwh_std = None
        self.duration = None
        self.duration_list = []
        self.duration_mean = None
        self.duration_std = None
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def write_log(self):
        """
        writes log for inference experiment
        :return:
        """
        # identify next running number of inference experiments
        dir_inference_logs = os.listdir('./results/' + self.dataset + '/')
        all_logs_for_experiment = [k for k in dir_inference_logs if k.startswith(self.counter)]
        # set log file accordingly
        logfile = ('./results/' + self.dataset + '/' + self.counter + '_' +
                   str(len(all_logs_for_experiment)) + '_' + self.timestamp + '.csv')
        # write log
        with open(logfile, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=INFERENCE_LOG_FIELDNAMES)
            writer.writeheader()
            writer.writerow({'users': self.user_list, 'results': self.result_list,
                             'scores': self.scores_list, 'kwh_consumed': self.kwh_consumed,
                             'duration': self.duration})

    def evaluate_topk(self, random_count=0, seed=42, runs=1, passes=1):
        """
        computes topk list and scores for all valid users of dataset
        alternatively, samples selectable number of users
        :param random_count: number of users to sample, 0=all
        :param seed: seed value for user sampling for reproducibility
        :param runs: how many identical queries in one CodeCarbon measurement
        :param passes: how many CodeCarbon measurements for mean and std dev
        :return:
        """

        # error detection
        if self.model_file is None:
            print('\nexperiment number does not exist.')
            return
        if not os.path.exists(self.model_file):
            print('\nmodel file does not exist.')
            return

        # load data and model
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=self.model_file)

        # get valid users
        # get mapping for user_id field, typically 0 is reserved for PAD
        user_token_mapping = dataset.field2token_id['user_id']
        # exclude the padding token if needed ('[PAD]')
        valid_user_tokens = [token for token in user_token_mapping.keys() if token != '[PAD]']

        # sample users if random_count has sufficient size, otherwise take all
        if 0 < random_count < len(valid_user_tokens):
            random.seed(seed)
            self.user_list = random.sample(valid_user_tokens, k=random_count)
        else:
            self.user_list = valid_user_tokens

        # get user uids
        uid_series = dataset.token2id(dataset.uid_field, self.user_list)
        # set emissions log file
        emissions_logfile = './results/' + self.dataset + '/inf_exp_' + self.counter + '_emissions_log.csv'

        # run passes times
        for _pass in range(passes):

            # warm up query (unmeasured)
            _, _ = full_sort_topk(uid_series, model, test_data, k=self.topk, device=config['device'])

            # execute model inference with CodeCarbon tracker
            with EmissionsTracker(project_name=self.model_file, output_file=emissions_logfile) as tracker:
                # run identical query runs times
                for _run in range(runs):
                    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=self.topk,
                                                               device=config['device'])

            # calculate energy consumption per run
            self.kwh_consumed = tracker.final_emissions_data.energy_consumed / runs
            # append to list
            self.kwh_list.append(self.kwh_consumed)
            # same for duration
            self.duration = tracker.final_emissions_data.duration / runs
            self.duration_list.append(self.duration)
            # save topk items and scores
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            self.result_list = external_item_list.tolist()
            self.scores_list = topk_score.tolist()
            # write log
            self.write_log()

        # compute energy and duration means and std dev
        self.kwh_mean = np.mean(self.kwh_list)
        self.kwh_std = np.std(self.kwh_list)
        self.duration_mean = np.mean(self.duration_list)
        self.duration_std = np.std(self.duration_list)


def run_training(dataset: str, model: str, metrics: str):
    """
    runs training experiment
    :return:
    """

    # error checking
    if dataset not in VALID_DATASETS:
        print('unknown dataset:', dataset)
        return
    if model not in VALID_MODELS:
        print('unknown model:', model)
        return
    else:
        model = VALID_MODELS[model]

    print(f"\n[TRAIN] mode, dataset={dataset}, model={model}, metrics={metrics}")

    # optional manual parameter override for testing
    """
    training_parameters = {
        'epochs': 50,
        'eval_step': 1,
        'stopping_step': 50,
        'valid_metric': 'MRR@10',
        'checkpoint_dir': './saved/' + dataset + '/',
        'save_dataset': True,
        'save_dataloaders': True,
        # 'metrics': ['MRR'],
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'MAP', 'Precision', 'GAUC', 'ItemCoverage',
                    'AveragePopularity', 'GiniIndex', 'ShannonEntropy', 'TailPercentage'],
        'topk': 10
    }
    """

    # setting to None triggers usage of per-algorithm preset parameters
    training_parameters = None
    # instantiate object and execute training
    experiment = TrainingExperiment(dataset_to_use=dataset, model_to_use=model)
    experiment.train_model(parameter_dict=training_parameters, metrics=metrics)

    print('\ntraining of algorithm', model, 'on dataset', dataset, 'completed.\n')

def run_evaluation(dataset: str, experiment: str):
    """
    runs evaluation experiment
    :return:
    """

    # error checking
    if dataset not in VALID_DATASETS:
        print('unknown dataset:', dataset)
        return

    print(f"\n[EVALUATE] mode, dataset={dataset}, experiment={experiment}")

    # manual override for testing, setting to None triggers set of 12 metrics
    # metrics = ['NDCG']
    metrics = None

    # instantiate object and execute evaluation
    evaluation = EvaluationExperiment(dataset_to_use=dataset, model_file=experiment)
    evaluation.evaluate_model(metrics=metrics)

    # print results
    print('\nvalidation set metrics:', evaluation.valid_result)
    print('\ntest set metrics:      ', evaluation.test_result, '\n')

def run_inference(dataset: str, experiment: str):
    """
    runs inference experiments
    :return:
    """

    # error checking
    if dataset not in VALID_DATASETS:
        print('unknown dataset:', dataset)
        return

    print(f"\n[INFERENCE] mode, dataset={dataset}, experiment={experiment}")

    # instantiate object and execute inference query
    # use k=10 for top-k, 10 runs per measurement, 10 passes for mean and std dev
    inference = InferenceExperiment(dataset_to_use=dataset, model_file=experiment, topk=10)
    inference.evaluate_topk(random_count=0, seed=42, runs=10, passes=10)

    # print results
    print('\nenergies (per query):', inference.kwh_list)
    print('energies mean:       ', inference.kwh_mean)
    print('energies std:        ', inference.kwh_std)
    print('\ndurations (per query):', inference.duration_list)
    print('durations mean:       ', inference.duration_mean)
    print('durations std:        ', inference.duration_std, '\n')

def non_empty_string(value: str) -> str:
    """
    helper function
    """
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return value

def main():
    # force every argument to lowercase
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0]] + [arg.lower() for arg in sys.argv[1:]]

    parser = argparse.ArgumentParser(
        description="Run reproducibility study: train, evaluate, or inference."
    )
    subparsers = parser.add_subparsers(
        title="modes",
        dest="mode",
        required=True,
        help="Select one of: train, evaluate, inference"
    )

    # TRAIN subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model: requires --dataset, --model, --metrics"
    )
    train_parser.add_argument(
        "--dataset",
        type=non_empty_string,
        required=True,
        help="Name of the dataset to use"
    )
    train_parser.add_argument(
        "--model",
        type=non_empty_string,
        required=True,
        help="Name of the model to use"
    )
    train_parser.add_argument(
        "--metrics",
        type=str,
        choices=["one", "all"],
        required=True,
        help="Whether to compute 'one' metric or 'all' metrics"
    )
    train_parser.set_defaults(func=lambda args: run_training(
        args.dataset, args.model, args.metrics
    ))

    # EVALUATE subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an experiment: requires --dataset, --experiment"
    )
    eval_parser.add_argument(
        "--dataset",
        type=non_empty_string,
        required=True,
        help="Name of the dataset to evaluate on"
    )
    eval_parser.add_argument(
        "--experiment",
        type=non_empty_string,
        required=True,
        help="Experiment number or path to model"
    )
    eval_parser.set_defaults(func=lambda args: run_evaluation(
        args.dataset, args.experiment
    ))

    # INFERENCE subcommand
    infer_parser = subparsers.add_parser(
        "inference",
        help="Run inference: requires --dataset, --experiment"
    )
    infer_parser.add_argument(
        "--dataset",
        type=non_empty_string,
        required=True,
        help="Name of the dataset to run inference on"
    )
    infer_parser.add_argument(
        "--experiment",
        type=non_empty_string,
        required=True,
        help="Experiment number or path to model"
    )
    infer_parser.set_defaults(func=lambda args: run_inference(
        args.dataset, args.experiment
    ))

    # if no args, show help and exit cleanly
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # allow “?” as a synonym for help
    if any(arg == "?" for arg in sys.argv[1:]):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
