import sys
sys.path.append("../../")

from train_model import train

hyperparameter={
    "num_nearest":5,
    "sigma":10,
    "learning_rate":0.0019,
    "batch_size":250,
    "num_neuron":60,
    "num_layers":3,
    "size_embedded":50,
    "num_nearest_geo":5,
    "num_nearest_eucli":5,
    "id_dataset":'poa',
    "epochs":20,
    "optimier":'adam',
    "validation_split":0.1,
    "label":'asi_fc',
    "early_stopping": False,
    "graph_label":'matrix',
    "use_masking": True, # new param. Set num_nearest, num_nearest_geo, num_nearest_eucli the same value
    "mask_dist_threshold": 0.1 # new param. Required if use_masking is True
}

spatial = train(**hyperparameter)

dataset,\
result,\
fit,\
embedded_train,\
embedded_test,\
predict_regression_train,\
predict_regression_test = spatial()

print('################# Test ##########################')
print('MALE test:.... {}'.format(result[0]))
print('RMSE test:.... {}'.format(result[1]))
print('MAPE test:.... {}'.format(result[2]))
print('################# Train ##########################')
print('MALE train:.... {}'.format(result[3]))
print('RMSE train:.... {}'.format(result[4]))
print('MAPE train:.... {}'.format(result[5]))