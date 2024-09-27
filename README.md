# House Price Prediction using Machine Learning
This project aims to predict house prices using various machine learning techniques. By analyzing multiple features such as crime rate, number of rooms, age of the property, and distance from various amenities, we build a model that can estimate the price of a house in a given area.
This project leverages XGBoost algorithm and Boston Housing Dataset to analyze key factors like location, size, crime rate, and air quality, providing reliable price estimates for potential homebuyers and investors.

# Project Overview
The project involves training a machine learning model to predict house prices based on historical data. We use regression algorithms like XGBoost to fit a model to the data, and the model is capable of predicting the price for new input data.


# Dataset

- House_Price.csv Dataset
- 507 rows and 19 columns
- The dataset used for this project contains the following columns:
price: House price.
crime_rate: Crime Rate (scale of 1-10).
resid_area: Residential Area (sqft).
air_qual: Air quality index.
room_num: Number of rooms in the house.
age: Age of the property (in years).
dist1: Distance to city center(km).
dist2: Distance to Public Transport (km).
dist3: Distance to School (km).
dist4: Distance to Hospital (km).
teachers: Number of teachers in nearby Schools.
poor_prop: Proportion of poor Families in the Area (%).
airport: Distance to the nearest airport (in kilometers).
n_hos_beds: Number of Hospital Beds in nearby Hospital.
n_hot_rooms: Number of Hotel Rooms in Area.
waterbody: Proximity to a waterbody (scale of 1-5).
rainfall: Proximity to Waterbody (1-5 scale).
bus_ter: Distance to the nearest bus terminal (km).
parks: Number of parks in the area.

# Features
Feature Engineering: Basic preprocessing, including handling missing values and scaling.
Modeling: Trained using XGBoost, a high-performance decision-tree-based algorithm.
Evaluation: Model evaluation using performance metrics like R² score and Mean Squared Error (MSE).


# Dependencies/Requirements

To run the project, you need the following libraries:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn


# Machine Learning Model

- XGBoost model for prediction
- Hyperparameter tuning using GridSearchCV
- XGBoost Regressor
- Hyperparameter Tuning: GridSearchCV
- Training Data: Boston Housing Dataset
- Pipeline:
    1. Data Preprocessing:
        - Handling missing values (Imputer)
        - Feature scaling (StandardScaler)
        - Encoding categorical variables (OneHotEncoder)
    2. Feature Selection:
        - Correlation analysis
        - Recursive Feature Elimination (RFE)
    3. Model:
        - XGBoost Regressor
        - Hyperparameters:
            - Learning Rate: 0.1
            - Max Depth: 5
            - Number of Estimators: 100
    4. Hyperparameter Tuning:
        - GridSearchCV

- Features: Crime Rate, Residential Area, Air Quality Index, Number of Rooms, Age of Property, Distances to City Center, Public Transport, School, and Hospital, Number of Teachers, Proportion of Poor Families, Distance to Airport, Number of Hospital Beds, Number of Hotel Rooms, and Proximity to Waterbody


  
# Installation

1. Clone the repository: git clone
2. Install required libraries: pip install -r requirements.txt
3. Run Jupyter Notebook: jupyter notebook


# Usage

1. Open House Price Prediction Using Machine Learning.ipynb in Jupyter Notebook.
2. Run all cells to train the model and make predictions.


# Results:

The model achieved the following performance metrics on the testing set:

- Mean Squared Error (MSE): 25.954592013593974
- Root Mean Squared Error (RMSE): 5.09456494841257
- R-Squared (R2): 0.6480459399756863
- Mean Absolute Error (MAE): 3.35022741873567



# Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

# License

MIT License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments:

- Boston Housing Dataset providers
- XGBoost library developers
- Scikit-learn library developers

# Authors

Lakshman Chaudhary
- GitHub: https://github.com/lakshman562000
- Linkedin: https://www.linkedin.com/in/lakshman-chaudhary-4532061ba/


# Contact

For questions, suggestions, or contributions, please contact mailto: lakshman562000@gmail.com

