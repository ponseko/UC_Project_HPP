hyperparameter={
    "num_nearest":60,
    "sigma":10,
    "learning_rate":0.008,
    "batch_size":250,
    "num_neuron":15,
    "num_layers":2,
    "size_embedded":50,
    "num_nearest_geo":30,
    "num_nearest_eucli":30,
    "id_dataset":'poa',
    "epochs":300,
    "optimier":'adam',
    "validation_split":0.1,
    "label":'asi_poa',
    "early_stopping": False,
    "graph_label":'matrix',
    "use_masking": False, # false for original
    "mask_dist_threshold": 0.1 # not used if use_masking is False
}