import sys
sys.path.append("../../")

import pandas as pd

from train_model import train

from notebooks.fc.hyperparams import hyperparameter as fc_hyperparams
from notebooks.kc.hyperparams import hyperparameter as kc_hyperparameter
from notebooks.poa.hyperparams import hyperparameter as poa_hyperparameter
from notebooks.sp.hyperparams import hyperparameter as sp_hyperparameter

envs = {
    "kc": {
        "hyperparameter": kc_hyperparameter,
        "max_neighbours": [50, 50, 50, 50, 50, 100, 150, 250], # efficient value for each of the thresholds
        "sequence": "300" # dataset number of neihbours to use, just for efficiency
    },
    "poa": {
        "hyperparameter": poa_hyperparameter,
        "max_neighbours": [50, 100, 100, 150, 250, 450, 700, 950],
        "sequence": "1200" 
    },
    "sp": {
        "hyperparameter": sp_hyperparameter,
        "max_neighbours": [50, 50, 100, 150, 350, 650, 1000, 1250],
        "sequence": "2400"
    },
    "fc": {
        "hyperparameter": fc_hyperparams,
        "max_neighbours": [50, 50, 150, 250, 500, 1050, 1550, 2250],
        "sequence": "2400"
    }
}

def train_and_save(db_name, hyperparameter, df, iteration):
    spatial = train(**hyperparameter)
    dataset, results, fit, embedded_train, embedded_test, predict_regression_train, predict_regression_test = spatial(SEQUENCE=envs[db_name]['sequence'])
    # add results to dataframe
    res = {
        "Dataset": db_name, 
        "iteration": iteration,
        "isMasked": hyperparameter["use_masking"], 
        "threshold": hyperparameter["mask_dist_threshold"],
        "MALE_test": results[0], 
        "RMSE_test": results[1], 
        "MAPE_test": results[2], 
        "MALE_train": results[3], 
        "RMSE_train": results[4],
        "MAPE_train": results[5]
    }
    df = pd.concat([df, pd.DataFrame([res])])
    df.to_csv(f"results_{db_name}.csv")
    return df


if __name__ == "__main__":
    df = pd.DataFrame([] ,columns=["Dataset", "iteration", "isMasked", "threshold", "MALE_test", "RMSE_test", "MAPE_test", "MALE_train", "RMSE_train", "MAPE_train"]) 

    thresholds = [0.01, 0.05, 0.1, 0.2, 0.35, 0.6, 0.8, 1.0]
    env_name = sys.argv[1]

    assert len(thresholds) == len(envs[env_name]["max_neighbours"])

    ITERATIONS = 10

    for i in range(ITERATIONS):
        print("Training on {}".format(env_name))
        
        params = envs[env_name]["hyperparameter"]
        params["epochs"] = 5
        df = train_and_save(env_name, params, df, iteration=i+1) # original run

        params["use_masking"] = True
        for threshold, max_neighbours in zip(thresholds, envs[env_name]["max_neighbours"]):
            print(max_neighbours)
            params["num_nearest"] = max_neighbours
            params["num_nearest_geo"] = max_neighbours
            params["num_nearest_eucli"] = max_neighbours
            params["mask_dist_threshold"] = threshold
            
            print("Training on {} with threshold {} and max_neighbours {}".format(env_name, params["mask_dist_threshold"], params["num_nearest"]))

            df = train_and_save(env_name, params, df, iteration=i+1)
