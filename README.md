# Internship-Project
This project is related to calculate the optimal price of flight ticket and optimal time also. 

## Table of Contents

  * Overview
  * Installation
  * Deployment
  * Directory Tree
  * Technologies Used
  * Future scope of project

## Overview
This is a Flask web app which predicts optimal time to book a flight.

## Objective
Objective is to predict the best optimal price of flight ticket and best optimal time to buy the ticket before to start the journey. So, here considering all the criteria such as number of stops, type of cabin class, name of airline company, cities and so on. So that a person can get the best deal from available.

## Installation
The Code is written in Python 3.9.12. If you don't have Python installed you can find it [here](https://https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```bash
    pip install -r requirements.txt
```

## Deployment
There are many cloud platforms where a machine learning model can deploy such as Flask, Django, Streamlit and all. But here I chose Heroku. 
To make an account on Heroku is completed free. Login there in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

## Directory Tree
```bash
├── static 
│   ├── css
├── template
│   ├── home.html
├── Procfile
├── README.md
├── app.py
├── MLpart.py
├── xgboostregressor.pkl
├── requirements.txt
├── runtime.txt

## Technologies Used
![madewith](https://raw.githubusercontent.com/Swapnil-Prajapati/optimal-flight-time-and-price-prediction/f35412262d3b5a9e1668c56f007abaf4efa9bcb7/Screenshots/madewith.svg)

![Flask](https://github.com/Swapnil-Prajapati/optimal-flight-time-and-price-prediction/blob/main/Screenshots/flask.png?raw=true) 
![Gunicorn](https://github.com/Swapnil-Prajapati/optimal-flight-time-and-price-prediction/blob/main/Screenshots/gunicorn.png?raw=true)
![Sklearn](https://github.com/Swapnil-Prajapati/optimal-flight-time-and-price-prediction/blob/main/Screenshots/scikit.png?raw=true)

## Future scope of project
- According to given optimal time build flight reccomender system
- Improvement on app.py and front-end.
- Optimizing Machine Learning Model and look for better substitute in future.
