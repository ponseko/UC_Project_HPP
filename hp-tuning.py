import keras_tuner as kt

import numpy as np
import os
from config import PATH

import asi.input_dataset as geds
import utils.utils as u
import utils.utilsdeep as ud

from tensorflow.keras.optimizers import Adam
from asi.input_phenomenon import getinput
from asi.input_neighborhood import getcontext
from asi.interpolation import Interpolation as interpolation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError

def build_model(
    hp: kt.HyperParameters,
    shape_input_phe, 
    shape_input_phe_w_lat_long, 
    shape_context_geo_target_dist, 
    shape_context_struc_w_lat_long, 
    shape_context_struc_target
):  
    clear_session()

    # Hyperparameters
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    _sigma = hp.Float('sigma', min_value=0.1, max_value=15, sampling='linear')
    sigma = [_sigma, _sigma]
    num_layers = hp.Int('num_layers', min_value=1, max_value=8, sampling='linear')
    num_neuron = hp.Int('num_neuron', min_value=2, max_value=128, sampling='log')

    # Fixed parameters
    num_nearest = num_nearest_eucli = num_nearest_geo = 1200
    size_embedded = 50
    use_masking = True

    optimizer = Adam(learning_rate=learning_rate)
    input_phenomenon, \
    input_phe_w_lat_long, \
    repeat_phe_w_lat_long = getinput(shape_input_phe, shape_input_phe_w_lat_long, num_nearest)

    context_geo_target_dist, \
    context_struc_w_lat_long, \
    context_struc_eucli_target, \
    dist_geo, \
    dist_eucli, \
    mask = getcontext(shape_context_struc_target,
                      shape_context_struc_w_lat_long,
                      shape_context_geo_target_dist, num_nearest,
                      True, True, num_nearest_geo, num_nearest_eucli,
                      use_masking=use_masking)
    
    embbeding = interpolation(
            geointerpolation='simple asi',
            shape_input_phenomenon=shape_input_phe,
            shape_input_phenomenon_eucl=shape_input_phe_w_lat_long,
            input_phenomenon=input_phenomenon,
            context_struc_eucli_target=context_struc_eucli_target,
            context_geo_target_dist=context_geo_target_dist,
            type_compat_funct_eucli='identity',
            type_compat_funct_geo='kernel_gaussiano',
            num_features_extras_struct=2,
            num_features_extras_geo=2,
            cal_dist_struct=False,
            cal_dist_geo=False,
            graph_label='graph_label',
            dist_eucli=dist_eucli,
            dist_geo=dist_geo,
            sigma=sigma,
            num_nearest=num_nearest,
            num_nearest_geo=num_nearest_geo,
            num_nearest_eucli=num_nearest_eucli,
            num_neuron=num_neuron,
            num_layers=num_layers,
            size_embedded=size_embedded,
            input_phe_w_lat_long=input_phe_w_lat_long,
            geo=True,
            euclidean=True,
            activation='elu',
            mask=mask
        ).run()
    
    # output
    main_output = Dense(1, activation='linear', name='main_output')(embbeding)
    inputs = [input_phenomenon, context_geo_target_dist,
                                  context_struc_eucli_target,
                                  dist_geo, dist_eucli]
    if mask != None:
        inputs.append(mask)
    
    model = Model(inputs=inputs, outputs=[main_output])
    # Compile the model
    model.compile(optimizer=optimizer, loss='mae', metrics=[RootMeanSquaredError()])

    return model

def run_tuning(dataset):
        mask_dist_threshold = 0.1
        num_nearest = num_nearest_eucli = num_nearest_geo = 1200

        train_x_p, test_x_p, \
        train_x_d, test_x_d, \
        train_x_g, test_x_g, \
        train_x_e, test_x_e, \
        train_mask, test_mask, \
        X_train, X_test, y_train,\
        y_test, y_train_scale = geds.Geds(id_dataset=dataset,
                                                     num_nearest=num_nearest,
                                                     geo=True,
                                                     euclidean=True, scale=True,
                                                     sequence='',
                                                     input_target_context=True,
                                                     input_dist_context_geo=True,
                                                     input_dist_context_eucl=True,
                                                     scale_euclidean=True,
                                                     scale_geo=True,
                                                     mask_dist_threshold=mask_dist_threshold)()
        shape_input_phe = X_train.shape[1]
        shape_input_phe_w_lat_long = X_train[:, 2:].shape[1]
        shape_context_geo_target_dist = train_x_d.shape[2]
        shape_context_struc_w_lat_long = X_train[:, 2:].shape[1]
        shape_context_struc_target = train_x_p.shape[2]

        # Construct training features
        inp_train = [X_train[:, :], train_x_d[:, :num_nearest_geo, :],
                     train_x_p[:, :num_nearest_eucli, :], train_x_g[:, :num_nearest_geo],
                     train_x_e[:, :num_nearest_eucli]]
        
        inp_train.append(train_mask[:, :])


        # Construct testing features
        inp_test = [X_test[:, :], test_x_d[:, :num_nearest_geo, :],
                     test_x_p[:, :num_nearest_eucli, :], test_x_g[:, :num_nearest_geo],
                     test_x_e[:, :num_nearest_eucli]]
        inp_test.append(test_mask[:, :])
        
        _build_asi_model = lambda x: build_model(x, shape_input_phe, shape_input_phe_w_lat_long, shape_context_geo_target_dist, shape_context_struc_w_lat_long, shape_context_struc_target)

        # Tuner
        tuner = kt.BayesianOptimization(
            _build_asi_model,
            objective='val_loss',
            max_trials=50,
            executions_per_trial=3,
            directory='tuning',
            project_name=f'asi_tuning_{dataset}',
        )

        tuner.search(inp_train, y_train, epochs=200, batch_size=256, validation_split=0.1, validation_data=(inp_test, y_test))
        tuner.results_summary()
        print(tuner.results_summary())

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    run_tuning(args.dataset)
