import sys
sys.path.append("../../")

from train_model import train
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
import os
from datetime import datetime

def run_model(config):
    spatial = train(**config)
    _, results, _, _, _, _, _ = spatial()
    # Return RMSE on test set
    return results[1]


class UniformRange():
    
    def __init__(self, low, high, is_integer=False):
        self.low = low
        self.high = high
        self.is_int = is_integer

    def sample(self):
        val = np.random.uniform(self.low, self.high)
        if self.is_int:
            return int(round(val))
        
        return val

class LogUniformRange():
        
    def __init__(self, low, high, is_integer=False):
        self.low = low
        self.high = high
        self.is_int = is_integer

    def sample(self):
        val = np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
        if self.is_int:
            return int(round(val))
        
        return val

class DiscreteRange():
        
    def __init__(self, values):
        self.values = values

    def sample(self):
        return np.random.choice(self.values)

class RandomSearch():

    def __init__(self, config_ranges, fixed_params=dict(), seed=42, save_dir="./output/hyperparameters/nl/", save_results=True):
        self.config_ranges = config_ranges
        self.fixed_params = fixed_params
        self.seed = seed
        self.tried_configs = []
        self.scores = []
        self.save_dir = save_dir
        self.save_results = save_results

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def run(self, num_samples, reset=True, filename=None):
        print(f"Running random search with {num_samples} samples")
        if reset:
            self.tried_configs = []
            self.scores = []
            
            if self.save_results and filename is None:
                raise ValueError("Must provide filename if save_results is True")

            if self.save_results and filename is not None:
                open(os.path.join(self.save_dir, filename), 'w+').close()
        
        
        
        configs = [self._sample_config() for _ in range(num_samples)]
        for i, config in enumerate(configs):
            print("============================")
            print(f"Running config {i+1}/{num_samples}")
            print("============================")
            score = run_model(config)
            with open(os.path.join(self.save_dir, filename), 'a+') as f:
                f.write(f"{score}, {config}\n")
            
            if score < min(self.scores, default=float("inf")):
                print("New best score!")
                print(f"Score: {score} Config: {config}")

            self.scores.append(score)
            self.tried_configs.append(config)
        
    def _sample_config(self):
        config = deepcopy(self.fixed_params)
        for key, range in self.config_ranges.items():
            config[key] = range.sample()
        
        config["num_nearest_geo"] = config["num_nearest"]
        config["num_nearest_eucli"] = config["num_nearest"]
        return config
    
    def _get_best_config(self):
        return self.tried_configs[np.argmin(self.scores)], np.min(self.scores)
        

def tune_old_model(dataset: str, num_samples: int = 10, seed: int = 42):
    config_settings = {
        'num_nearest': UniformRange(1, 100, is_integer=True),
        # 'num_nearest_geo': UniformRange(1, 100, is_integer=True),
        # 'num_nearest_eucli': UniformRange(1, 100, is_integer=True), # These get assigned to num_nearest
        'sigma': UniformRange(1, 15),
        'learning_rate': LogUniformRange(0.0001, 0.01),
        'num_neuron': LogUniformRange(2, 128, is_integer=True),
        'num_layers': DiscreteRange([1, 2, 3, 4, 5]),        
    }

    fixed_params = {
        'batch_size': 250,
        'size_embedded': 50,
        'id_dataset': dataset,
        'epochs': 100,
        'optimier': 'adam',
        'validation_split': 0.1,
        'label': f'asi_{dataset}',
        'early_stopping': False,
        'graph_label': 'matrix',
        'use_masking': False,
        'mask_dist_threshold': 0.1,
    }

    search = RandomSearch(config_settings, fixed_params, seed=seed, save_dir="./output/hyperparameters/nl", save_results=True)
    search.run(num_samples, reset=True, filename=f"results_old_model_{dataset}_{seed}.csv")
    best_config, best_score = search._get_best_config()
    print(f"Best config: {best_config}")
    print(f"Best score: {best_score}")
    search.save_results(f"./output/hyperparameters/nl/results_{dataset}_{seed}.csv")

def tune_new_model(dataset: str, num_samples: int = 10, seed: int = 42):
    config_settings = {
        'sigma': UniformRange(1, 15),
        'learning_rate': LogUniformRange(0.0001, 0.01),
        'num_neuron': LogUniformRange(2, 128, is_integer=True),
        'num_layers': DiscreteRange([1, 2, 3, 4, 5]),
        'mask_dist_threshold': DiscreteRange([0.1, 1.0, 2.5, 5.0, 10.0, 25.0, 35.0, 50.0]),
    }

    fixed_params = {
        'num_nearest': 2400,
        'num_nearest_geo': 2400,
        'num_nearest_eucli': 2400,
        'batch_size': 250,
        'size_embedded': 50,
        'id_dataset': dataset,
        'epochs': 100,
        'optimier': 'adam',
        'validation_split': 0.1,
        'label': f'asi_{dataset}',
        'early_stopping': False,
        'graph_label': 'matrix',
        'use_masking': True,
    }

    search = RandomSearch(config_settings, fixed_params, seed=seed, save_dir="./output/hyperparameters/nl", save_results=True)
    search.run(num_samples, reset=True, filename=f"results_new_model_{dataset}_{seed}.csv")
    best_config, best_score = search._get_best_config()
    print(f"Best config: {best_config}")
    print(f"Best score: {best_score}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nl')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--new_model', action='store_true')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    if args.new_model:
        tune_new_model(args.dataset, num_samples=args.num_samples, seed=args.seed)
    else:
        tune_old_model(args.dataset, num_samples=args.num_samples, seed=args.seed)