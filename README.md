# UC_Project_HPP

UC Project House Price Prediction - adaptation of https://github.com/darniton/ASI

For the main experiments, it is required to download larger data files from [here](https://leidenuniv1-my.sharepoint.com/:f:/g/personal/s1861581_vuw_leidenuniv_nl/EhQCxNUxMT1IkC6_yECd0k8BcwKgQ5O29v6zQCG3zxiIHA) and place them in their respective datasets folder.

The requirements on some of the datasets (particularly fc, may require a very large amount of RAM).

Our main experiments can be run with `python experiments.py [dataset]`

dataset can be any of:

-   fc
-   kc
-   poa
-   sp
-   nl

The results folder contains results from our experiments along with a notebook of visualizing them.

# ASI

Attention-Based Spatial Interpolation for House Price Prediction

### Requirements

-   python 3.8.10
-   tensorflow (>=2.5.0)

## Data

-   Data: a numpy saved file (.npz) containing:
    -   data['dist_eucli']
    -   data['dist_geo']
    -   data['idx_eucli']
    -   data['idx_geo']
    -   data['X_train']
    -   data['X_test']
    -   data['y_train']
    -   data['y_test']

## Installation

To install as a module:

```
$ conda create -n asi python=3.8.10
$ conda activate asi
$ git clone https://github.com/darniton/ASI
$ cd ASI
$ pip install -r requirements.txt
$ jupyter notebook
```

## Replicating experiments

To replicate the experimental results:

-   ./notebooks: One notebook for each dataset
