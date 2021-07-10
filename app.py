import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
# Deserialization using Pickle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text='Quality: {}'.format(str(output)))

if __name__ == '__main__':
    app.run(port=5000, debug=True)

