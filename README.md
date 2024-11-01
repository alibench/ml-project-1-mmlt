<div id="top"></div>

## About The Project
This project was developed by Sarra Chaabane, Aicha Masmoudi and Ali Benchekroun as part of the Machine Learning course at EPFL. The goal was to predict the health status of individuals based on a dataset of 300,000 individuals. The dataset contains 358 features and the target variable is binary. 

### Built With

* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.7 or higher
* Numpy

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/CS-433/ml-project-1-mmlt/tree/main
    ```
2. Download the dataset and add it to the `data/dataset` folder (<a href="https://www.cdc.gov/brfss/annual_data/annual_2015.html">CS433 Machine Learning</a>.)

### Running

Open ```run.ipynb``` and run all the cells to create the submission file (.csv)

<!-- PROJECT STRUCTURE -->
## Project Structure

The project is structured as follows:

```
├── data
    ├── data_to_release
        ├── x_train.csv
        ├── x_test.csv
        ├── y_train.csv
    
|__ predictions
│   ├── submission.csv

|__ helpers.py
|__ implementations.py
|__ metrics.py
|__ processing.py
|__ run.ipynb
|__ training.ipynb
|__ utils.py
```
<!-- Content of the files -->
## Content of the files

* `helpers.py`: Contains helper functions to load the data and create csv submissions.
* `implementations.py`: Contains the implementation of the six required algorithms.
* `metrics.py`: Contains the metrics used to evaluate the models.
* `processing.py`: Contains the functions used to preprocess the data.
* `run.ipynb`: Main file; Jupyter notebook to run the project.
* `training.ipynb`: Jupyter notebook with the function used to train the models.
* `utils.py`: Contains utility functions used in the project.

<p align="right">(<a href="#top">back to top</a>)</p>


  

