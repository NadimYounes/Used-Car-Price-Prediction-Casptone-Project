from flask import Flask
from flask import render_template,request
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Create an app object
app = Flask(__name__)

#Whenever this is called it will activate the function
@app.route('/')
def home(): # initialize a web app with flask
	return render_template('index.html') # template for user to add input to (front page)

# please note i dont use the saved model but recall a new one because i dont include country as a feature in my web app
#Load CSV
span_new = pd.read_csv('Span_new.csv')
span_X = span_new[['make','model','age','gear_type','fuel_type','power','kms']]
# Dummy Categorical Variables
span_X = pd.concat([span_X,pd.get_dummies(span_X['make'],drop_first=True,prefix="Make")],axis=1)
span_X= pd.concat([span_X,pd.get_dummies(span_X['model'],drop_first=True,prefix="Model")],axis=1)
span_X = pd.concat([span_X,pd.get_dummies(span_X['gear_type'],drop_first=True,prefix="Gear")],axis=1)
span_X = pd.concat([span_X,pd.get_dummies(span_X['fuel_type'],drop_first=True,prefix="Fuel")],axis=1)
features_final = span_X.drop(columns=['make','model','gear_type','fuel_type'])
X_train, X_test, y_train, y_test = train_test_split(features_final, span_new['price'], test_size=0.33, random_state=42)

#Instatiate and Fit Model
rand_est = RandomForestRegressor()
rand_est.fit(X_train,y_train)

#FUNCTION that converts user's input into a vector that can be used by the model (29x features)
def prep_features(result):
    vector = np.zeros(292) # Number of features in my dataset, dependent variables

    makez = 'Make_'+ result['Make'] #Make it return the column name e.g Make_BMW
    modelz = 'Model_'+ result['Model']
    fuelz = 'Fuel_' + result['fuel_type']
    gearz = 'Gear_'+ result['gear_type']

    make_index = features_final.columns.get_loc(str(makez)) # Get the index of the column using df.columns.get_loc()
    model_index = features_final.columns.get_loc(str(modelz))
    fuel_index = features_final.columns.get_loc(str(fuelz))
    gear_index = features_final.columns.get_loc(str(gearz))

    vector[0] = result['age'] # Input values into your np.zeros vector
    vector[1] = result['power']
    vector[2] = result['kms']
    vector[make_index]= 1 # this will put a 1 at the right position ( depending on what make is selected)
    vector[model_index]= 1
    vector[fuel_index]= 1
    vector[gear_index]= 1
    return vector


@app.route('/predictprice', methods=['POST','GET']) # this is the actual post request which will take the inputs run the model and hopefully return a prediction
def predict_price():
	print(request.__dict__) # test to check what it is being inputted into request
	if request.method=='POST':
	 	#Storage method for data (requests the web server to accept data)
		cars=request.form
		print(request.form.__dict__)# this form will take the inputs
		age=cars['age']
		power=cars['power']
		kms=cars['kms']
		make=cars['make']
		model=cars['model']
		gear_type=cars['gear_type']
		fuel_type=cars['fuel_type']

		# the function from above will convert this result dict into a vector of 292 values
		result = {'age':age, 'power':power, 'kms':kms,'Make':make, 'Model':model,'gear_type':gear_type,'fuel_type':fuel_type}

		print("hey")
		test = prep_features(result)
		print(test)
		prediction = rand_est.predict([test])
		# calls result.html which displays the output from the random forest
	return render_template('result.html',prediction=prediction) # returns prediction

app.run(debug=True)
