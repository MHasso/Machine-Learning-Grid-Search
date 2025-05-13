# Machine Learning Grid Search

This repository demonstrates the use of **Grid Search** for hyperparameter tuning in machine learning models using `scikit-learn`. It includes an example of building regression pipelines with preprocessing, training, and evaluation using random forest models.

## Features

- Regression modeling using:
  - `RandomForestRegressor`
- Data preprocessing with pipelines:
  - Scaling numeric features
  - One-hot encoding categorical features
- Hyperparameter tuning with `GridSearchCV`
- Performance evaluation using Mean Absolute Error (MAE)

## Project Structure

```
├── data/                    # (Optional) Place your dataset here
├── data_loader.py           # load the data and perform preprocessing
├── grid_search.py           # create and train the grid search to get the best params
├── ml_model.py              # create a random forest model and train it to compare the results with the grid search
├── main.py                  # Main script to run the pipeline and model training
├── requirements.txt         # Python dependencies
└── README.md                # Project overview
```

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/MHasso/Machine-Learning-Grid-Search.git
cd Machine-Learning-Grid-Search
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the script**
```bash
python main.py
```

## Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

Basic dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Notes

- You can update the dataset or modify the pipeline and grid parameters in `main.py` to suit your use case.
- The default models include `RandomForestRegressor`.

## License

This project is open-source and available under the [MIT License](LICENSE).
