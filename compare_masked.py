import sys
sys.path.append("../../")

import pandas as pd

from train_model import train

from notebooks.fc.hyperparams import hyperparameter as fc_hyperparams
from notebooks.kc.hyperparams import hyperparameter as kc_hyperparameter
from notebooks.poa.hyperparams import hyperparameter as poa_hyperparameter
from notebooks.sp.hyperparams import hyperparameter as sp_hyperparameter



def train_and_save(db_name, hyperparameter, df):
    spatial = train(**hyperparameter)
    dataset, results, fit, embedded_train, embedded_test, predict_regression_train, predict_regression_test = spatial()
    # add results to dataframe
    res = {
        "Dataset": db_name, 
        "isMasked": hyperparameter["use_masking"], 
        "MALE_test": results[0], 
        "RMSE_test": results[1], 
        "MAPE_test": results[2], 
        "MALE_train": results[3], 
        "RMSE_train": results[4],
        "MAPE_train": results[5]
    }
    df = pd.concat([df, pd.DataFrame([res])])

    df.to_csv("results.csv")

    return df

if __name__ == "__main__":
    df = pd.DataFrame([] ,columns=["Dataset", "isMasked", "MALE_test", "RMSE_test", "MAPE_test", "MALE_train", "RMSE_train", "MAPE_train"]) 

    df = train_and_save("FC", fc_hyperparams, df)
    df = train_and_save("KC", kc_hyperparameter, df)
    df = train_and_save("POA", poa_hyperparameter, df)
    df = train_and_save("SP", sp_hyperparameter, df)

    fc_hyperparams["use_masking"] = True
    kc_hyperparameter["use_masking"] = True
    poa_hyperparameter["use_masking"] = True
    sp_hyperparameter["use_masking"] = True

    fc_hyperparams["num_nearest_geo"] = 60
    fc_hyperparams["num_nearest_eucli"] = 60

    kc_hyperparameter["num_nearest_geo"] = 60
    kc_hyperparameter["num_nearest_eucli"] = 60

    poa_hyperparameter["num_nearest_geo"] = 60
    poa_hyperparameter["num_nearest_eucli"] = 60

    sp_hyperparameter["num_nearest_geo"] = 60
    sp_hyperparameter["num_nearest_eucli"] = 60

    df = train_and_save("FC", fc_hyperparams, df)
    df = train_and_save("KC", kc_hyperparameter, df)
    df = train_and_save("POA", poa_hyperparameter, df)
    df = train_and_save("SP", sp_hyperparameter, df)