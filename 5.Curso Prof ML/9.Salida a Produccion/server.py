import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    #Para hacer la prueba se la pasa la primera fila del archivo felicidad (features)
    prediction = model.predict(X_test.reshape(1,-1))  #Se pone el reshape para que lea una fila y las cols 
    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)

    #Para hacer la prueba abrir una webpage y poner localhost:8080/predict