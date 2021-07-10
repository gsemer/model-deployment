import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
# Deserialization using Pickle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['GET'])
def predict():
    fixed_acidity = request.args.get('fixed acidity')
    volatile_acidity = request.args.get('volatile acidity')
    citric_acid = request.args.get('citric acid')
    residual_sugar = request.args.get('residual sugar')
    chlorides = request.args.get('residual sugar')
    free_sulfur_dioxide = request.args.get('free sulfur dioxide')
    total_sulfur_dioxide = request.args.get('total sulfur dioxide')
    density = request.args.get('density')
    ph = request.args.get('pH')
    sulphates = request.args.get('sulphates')
    alcohol = request.args.get('alcohol')

    features = [fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                ph,
                sulphates,
                alcohol]

    float_features = [float(x) for x in features]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text='Quality: {}'.format(str(output)))

if __name__ == '__main__':
    app.run(port=5000, debug=True)

