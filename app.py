import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug import secure_filename
import pickle
import os


app = Flask(__name__)
model = pickle.load(open('randomForestRegressor.pkl','rb'))
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('home.html')
   

@app.route('/predict',methods = ['POST'])
def predict():

    fl = request.files['file']
    filename = secure_filename(fl.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    fl.save(file_path)
    with open(file_path) as f:
        file_content = f.read()
    
    print(type(file_content))
    print(file_content)

    txt_list = file_content.split(',')

    int_features = [float(x) for x in txt_list]
    print("type---:",type(int_features))
    print("values:",int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('home.html', prediction_text="prediction output -:{}".format(prediction[0]))



if __name__ == '__main__':
    app.run(debug=True)