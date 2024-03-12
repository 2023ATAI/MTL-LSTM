### Installation

Our works in [Python3.9.13](https://www.python.org/downloads/)
In order to use the code successfully, the following site-packages are required:

- pytorch 1.13.1
- pandas 1.4.4
- numpy 1.22.0
- scikit-learn 1.0.2
- scipy 1.7.3
- matplotlib 3.5.2
- xarray 2023.1.0
- netCDF4 1.6.2

### Prepare Config File

Usually, we use the config file in model training, testing and detailed analyzing.

The config file contains all necessary information, such as path,data,model, etc.

The config file of our work is `config.py`

### Train model

When the config is properly configured, run main.py to train the model.

### Display the results

Running postprocess.py will display the evaluation results of the model across various metrics.

### 





