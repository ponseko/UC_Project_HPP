hyperparameter={
    "num_nearest":60,
    "sigma":10,
    "learning_rate":0.0019,
    "batch_size":250,
    "num_neuron":60,
    "num_layers":3,
    "size_embedded":50,
    "num_nearest_geo":20,
    "num_nearest_eucli":20,
    "id_dataset":'fc',
    "epochs":300,
    "optimier":'adam',
    "validation_split":0.1,
    "label":'asi_fc',
    "early_stopping": False,
    "graph_label":'matrix',
    "use_masking": False, # false for original
    "mask_dist_threshold": 0.1 # not used if use_masking is False
}