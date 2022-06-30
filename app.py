from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
#from waitress import serve
import numpy as np

app = Flask(__name__)
model = pickle.load(open("xgboostregressor.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
                    
    if request.method == "POST":
        
        # Date_of_Journey
        date_dep = request.form["departure_time"]
        Dept_date = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        #Dept_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        #print("Journey Date : ",Dept_date, Dept_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        #print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["arrival_time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        #print("Arrival : ", Arrival_hour, Arrival_min)
        
        #Duration
        Duration_hour = abs(Arrival_hour - Dep_hour)
        Duration_min = abs(Arrival_min - Dep_min)
       
        # Total Stops
        stops = int(request.form["stops"])
        
        # Airline , Air India  0 not in column
        Airline= request.form["Airline"]
        if(Airline =='AirAsia'):
            AirAsia = 1
            IndiGo = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            
        elif(Airline =='IndiGo'):
            AirAsia = 0
            IndiGo = 1
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            
        elif(Airline =='Spicejet'):
            AirAsia = 0
            IndiGo = 0
            SpiceJet = 1
            Vistara = 0
            GoAir = 0
        
        elif(Airline =='Vistara'):
            AirAsia = 0
            IndiGo = 0
            SpiceJet = 0
            Vistara = 1
            GoAir = 0
            
        elif(Airline =='Go Air'):
            AirAsia = 0
            IndiGo = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 1
            
        else:
            AirAsia = 0
            IndiGo = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            
        
        # Departure city
        Dept_city = int(request.form["Dept_city"])
        
        # Arrival city
        arrival_city = int(request.form["arrival_city"])
        
        # Cabin B = 0 here
        Cabin = request.form["Cabin"]
        if(Cabin =='E'):
            E = 1
            PE = 0
        
        elif(Cabin=='PE'):
            E= 0
            PE = 1
        else:
            E = 0
            PE = 0
        
        
        # ['Dept_city', 'Dept_date', 'arrival_city', 'stops','Dep_hour', 'Dep_min',
       #'Arrival_hour', 'Arrival_min', 'Duration_hour', 'Duration_min',
       #'AirAsia', 'GoAir', 'IndiGo', 'Spicejet', 'Vistara', 'E', 'PE']
        
        data_to_predict = pd.DataFrame([[
            Dept_city,
            Dept_date,
            arrival_city,
            stops,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            Duration_hour,
            Duration_min,
            AirAsia,
            GoAir,
            IndiGo,
            SpiceJet,
            Vistara,
            E,
            PE
            ]],columns=['Dept_city', 'Dept_date', 'arrival_city', 'stops','Dep_hour', 'Dep_min',
                        'Arrival_hour', 'Arrival_min', 'Duration_hour', 'Duration_min',
                        'AirAsia', 'GoAir', 'IndiGo', 'Spicejet', 'Vistara', 'E', 'PE'])
       
        prediction = model.predict(data_to_predict)
        
        #output=round(prediction[0],2)

        #return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))
        
        price= np.round(prediction[0][1], 2)
        time = np.round(prediction[0][0], 2)
        
        minutes = time
        hours, minutes = divmod(minutes, 60)

        t = ("%02d:%02d"%(hours,minutes))

     
        return render_template('home.html',prediction_text= "Your optimal Flight Time: {0} And Price is Rs:{1:.2f} ".format(t,price))
    
        

    return render_template("home.html")

if __name__ == "__main__":
    #serve(app, host="127.0.0.1", port=8080)
    #app.run()
    app.run(debug=True, use_reloader=False)
