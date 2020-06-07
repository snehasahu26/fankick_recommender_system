import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
#from sklearn.preprocessing import LabelEncoder
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#from statsmodels.tsa.statespace.sarimax import SARIMAXResults
#import matplotlib.pyplot as plt
#import numpy as np 
import pandas as pd
#import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random


from fun_all import call_fun
#call_fun("59bd3c43b137a16aac26f79c")

app = Flask(__name__)



#result_out("59bd3c43b137a16aac26f79c")



'''
@app.route('/')
def home():
    return render_template('index.html')
'''

 
'''
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    
    out1=call_fun(int_features[0])
    
    
    
    
    #data="fafafa"

    return render_template('index.html',tables=[out1.to_html(classes='data', header="true")] )
   
  
####
'''


@app.route('/<string:name>')
def api_call(name:str):

    
    
    
    out1=call_fun(name)
    df = out1.to_dict()

    print(df)
    
    
    return jsonify(data=df),200
    
'''        
@app.route('/results',methods=['POST'])
def results():

 

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

 


    output = prediction[0]
    return jsonify(output)
''' 

if __name__ == "__main__":
    app.run(port=1024)
    
    
    
