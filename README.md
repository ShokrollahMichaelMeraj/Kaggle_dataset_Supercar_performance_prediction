# Kaggle_dataset_Supercar_0_to_60_Acceleration_prediction.


## Description

- End-to-end machine learning that predicts supercar 0–100 km/h acceleration times
using engineered physical ratios and a neural network trained on an updated
5,000+ Kaggle supercar dataset. Achieved **0.27–0.30s MAE**, improving over baseline
models (>2.6s MAE) through feature engineering, optimizer tuning, dataset
validation, and training stability techniques (gradient clipping).


## Problem Statement

- Accurately predicting 0–60 MPH acceleration is non-trivial due to the nonlinear
relationships between power, weight, drivetrain, transmission, and traction as well as other existing aspects.
Most public datasets contain noisy or incorrect labels, limiting model
performance. This project addresses both modeling and data-quality challenges.


## Feature Set:

-  Numerical features:

    - year (discounter or decay varible because older cars will have reduced horsepower)
    - initial horsepower
    - weight.
    - torque
    - torque to weight 
    - power to weight 

- Binary features

    - transmission 
        - DCT
        - Automatic
    
    - Drivetrain 
        - AWD
        - RWD

 
## Model Architecture

- Model type: Feedforward Neural Network (Regression)
- Framework: TensorFlow / Keras
- Final architecture:
  - Dense(64) → Dense(32) → Dense(32) → Dense(1)
  - ReLU activations
  - Dropout for regularization
- Loss function: Mean Absolute Error (MAE)
- Optimizer: SGD with momentum = 0.9
- Gradient clipping: clipnorm = 1.0


## system architecture:

```
supercar-0to60-prediction/



├── Frontend/
    │
    ├── data/
    │   ├── raw/                        # Original Kaggle CSVs (never modify these)
    │   └── processed/                  # Cleaned data ready for training
    │   
    │
    ├── notebooks/
    │   ├── 01_exploration.ipynb        # EDA (exploratory data analysis)
    │   ├── 02_preprocessing.ipynb      # Data cleaning & feature engineering
    │   └── 03_training.ipynb           # Model training experiments
    │   
    │
    ├── models/
    │   ├── feature_info.pkl
    │   ├── feature_names.pkl
    │   ├── nn_scaler.pkl
    │   ├── feature_names.pkl       
    │   └── nn_zero_to_sixty.keras 
    │
    ├── gui/
    │   └── app.py                      # Figma Make (UI) connected with Fast API (COMING SOON)
    │
    ├── reports/
    │   └── figures/ 
    │       ├── progression_notes.md    # Milestones, challenges, issues, solutions and results.
    │       └── Distribution_of_Supercar_0-60_vs_Torque.png              
    │   
    │
    ├── requirements.txt                # List of dependencies
    └── README.md                       # Project description
    


```



## Performance Comparison

| Model / Configuration                     | MAE (seconds) |
|------------------------------------------|---------------|
| Initial feature set (raw dataset)        | ~2.6          |
| Cleaned numerical features               | ~2.0          |
| + Ratio features                         | ~1.26         |
| + Corrected dataset                      | ~0.28         |
| Final model (stable, clipped gradients)  | **0.27–0.30** |


## Tech Stack

- Python, NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- Jupyter
- Figma Make / FastAPI (UI & inference)
- Supabase Backend
- Git, Kaggle



## How To Run:

### Getting Started:
1. Clone the Repository: ```git clone ... ```
2. Enter repository through visual studio code


### virtual environment set up

**For initial set up on MAC OS:**

```
python3 -m venv venv    

source venv/bin/activate 

pip3 install -r requirements.txt
```

**For initial set up on Windows:**

```
python3 -m venv venv    

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

venv\Scripts\Activate.ps1 # for powershell

venv\Scripts\activate.bat # for command prompt

pip install -r requirements.txt
```

**To exit the virtual environment:**
- kill terminal
- or run ``` deactivate ```


### Running the Notebook
1. Run ```jupyter notebook ``` in terminal, this will open up a web browser
2. select the notebook 
3. select desired notebook:
    - 01_exploration.ipynb        # EDA (exploratory data analysis)
    - 02_preprocessing.ipynb      # Data cleaning & feature engineering
    - 03_training.ipynb           # Model training experiments

4. run each cell to see our results

### Modifications and Use:

- You can use our saved models to predict your own 0-100 acceleration times by saving and running our models from the /model folder, or visit our User interface.




## User Interface:

- Generated using Figma Make, it enables users to make their own predictions.\
- Coming Soon!



