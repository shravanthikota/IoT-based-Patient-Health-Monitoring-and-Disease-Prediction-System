# import Flask class from the flask module
from flask import Flask, request
import pickle

# Libraries
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from flask import jsonify
import pyrebase
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Enabling CORS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Read Dataset
df = pd.read_csv('Dataset/cleaned_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:-1], df['num'], 
                                                    test_size = 0.33, random_state=44,
                                                   stratify= df['num'] )
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

## FIREBASE ##
config = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "",
  "storageBucket": ""
}

firebase = pyrebase.initialize_app(config)
db = firebase.database() 

## FIREBASE ##
data_label = ['No Heart Disease','Silent Ischemia','Coronary Artery Disease','Angina','Myocardial Infarction']

@app.route('/rf', methods=['GET', 'POST'])
def rf():
    # Get values
    global rf
    record_root_path = db.child("ali1")
    record_root = record_root_path.get()
    print(record_root.val())
    for record in record_root.each():
    	record_key = record.key()
    	record_val = record.val()
    	heart_rate = record_val['heart_rate']
    	spo2 = record_val['spo2']
    	temp = record_val['temp']
    	bp = record_val['s4']

    	print('Fetching data from firebase\n')
    	print('heart_rate='+ str(heart_rate))
    	print('spo2='+ str(spo2))
    	print('temp='+ str(temp))
    	print('bp='+ str(bp))

    	df = get_data(request.form,heart_rate,spo2,temp)
    	print(df)
    	scaled_df = scaler.transform(df)
    	class_prediced = rf.predict(scaled_df)
    	record_root_path.child("ali1").child(record.key()).update({"prediction": str(data_label[class_prediced[0]])})
    	print("Prediction is " + str(data_label[class_prediced[0]]))
    return (str(data_label[class_prediced[0]]))

# Routes for models Ends

def get_data(dict,heart_rate,spo2,temp):
    age = dict.get('age')
    sex = dict.get('sex')
    cp = dict.get('cp')
    trestbps = dict.get('trestbps')

    chol = dict.get('chol')
    fbs = dict.get('fbs')
    restecg = dict.get('restecg')
    thalach = dict.get('thalach')

    exang = dict.get('exang')
    oldpeak = dict.get('oldpeak')
    slope = dict.get('slope')
    ca = dict.get('ca')

    thal = dict.get('thal')

    data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang,oldpeak,slope,ca,thal,heart_rate,temp,spo2]).reshape(1, 16)
    df = pd.DataFrame(data)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach','exang','oldpeak','slope','ca','thal','heart_rate','temp','spo2']
    return (df)


def get_model():
    global rf

    rf_file = open('models/rf.pckl', 'rb')
    rf = pickle.load(rf_file)
    rf_file.close()

if __name__ == "__main__":
    print("**Starting Server...")

    # Call function that loads Model
    get_model()

# Run Server
get_model()
app.run()
