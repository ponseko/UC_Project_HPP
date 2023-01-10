hyperparameter = {
    "num_nearest":60,
    "sigma":10,
    "learning_rate":0.001,
    "batch_size":250,
    "num_neuron":60,
    "num_layers":3,
    "size_embedded":50,
    "num_nearest_geo":45,
    "num_nearest_eucli":45,
    "id_dataset":'nl',
    "epochs":300,
    "optimier":'adam',
    "validation_split":0.1,
    "label":'asi_nl',
    "early_stopping": False,
    "graph_label":'matrix',
    "use_masking": False, # false for original
    "mask_dist_threshold": 1.0 # not used if use_masking is False
}

# HYPERPARAMS OLD MODEL ON NL DATASET
{
    'batch_size': 250, 
    'size_embedded': 50, 
    'id_dataset': 'nl', 
    'epochs': 100, 
    'optimier': 'adam', 
    'validation_split': 0.1, 
    'label': 'asi_nl', 
    'early_stopping': False, 
    'graph_label': 'matrix', 
    'use_masking': False, 
    'mask_dist_threshold': 0.1, 
    'num_nearest': 57, 
    'sigma': 11.594341515575536, 
    'learning_rate': 0.00018976453845357367, 
    'num_neuron': 9, 
    'num_layers': 2,
     'num_nearest_geo': 57, 
     'num_nearest_eucli': 57
}

# HYPERPARAMS NEW MODEL ON NL DATASET (RMSE 146724.669972 during random search)
{
    'num_nearest': 2400, 
    'num_nearest_geo': 2400, 
    'num_nearest_eucli': 2400, 
    'batch_size': 250, 
    'size_embedded': 50, 
    'id_dataset': 'nl', 
    'epochs': 100, 
    'optimier': 'adam', 
    'validation_split': 0.1, 
    'label': 'asi_nl', 
    'early_stopping': False, 
    'graph_label': 'matrix', 
    'use_masking': True, 
    'sigma': 6.552137640157195, 
    'learning_rate': 0.0038521344844787636, 
    'num_neuron': 62, 
    'num_layers': 2, 
    'mask_dist_threshold': 2.5
}