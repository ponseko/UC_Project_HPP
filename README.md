# UC_Project_HPP

UC Project House Price Prediction - adaptation of Attention-Based Spatial Interpolation for House Price Prediction (https://github.com/darniton/ASI)

Our main experiments can be run with `python experiments.py [dataset]`

dataset can be any of:

-   fc
-   kc
-   poa
-   sp
-   nl

WARNING: running the full experiments, most notably on the fc dataset, requires a large amount of RAM.
The kc dataset experiments may be possible to run on a local computer with sufficient ram (16/32g).

The results folder contains results from our experiments along with a notebook of visualizing them.

### Data

For the main experiments, it is required to download larger data files from [here](https://leidenuniv1-my.sharepoint.com/:f:/g/personal/s1861581_vuw_leidenuniv_nl/EhQCxNUxMT1IkC6_yECd0k8BcwKgQ5O29v6zQCG3zxiIHA) and place them in their respective datasets folder.

### Requirements

-   python 3.8(.15)
-   other requirements listed in the requirements.txt
