# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:42:14 2023

@author: erick
"""

from flask import Flask
from flask import request
from flask import render_template
from joblib import load
import pandas as pd
  
# Le pasamos el control de la aplicacion a Flask
app = Flask(__name__)
  
modelo = load('modelo_suicidio.joblib')# Cargamos el modelo
#encoder = load('label_encoder.joblib')# Cargamos el label encoder
  
# Cuando carge la aplicacion en "/" mostraremos el index.html que esta en la carpeta templates
@app.route('/', methods=["GET"])
def main():
    return render_template('index1.html')
  
# Haremos la peticion a esta ruta para la prediccion
@app.route('/predecir', methods=["POST"])
def predecir():
    datos = request.json
    message = suicidePred(datos["input_edad"], datos["input_genero"],datos["input_depre"],datos["input_ansie"], 
                          datos["input_person"],datos["input_bipo"], datos["input_idensex"],datos["input_suici"],
                          datos["input_alcoh"], datos["input_sust"], datos["input_trasesq"], datos["input_esquiz"],
                          datos["input_obesi"], datos["input_dolor"], datos["input_maltra"],datos["input_abusex"])
    return message
  
# la funcion de prediccion
def suicidePred(input_edad,input_genero,input_depre,input_ansie,input_person,input_bipo,input_idensex,
                input_suici,input_alcoh,input_sust,input_trasesq,input_esquiz,input_obesi,input_dolor,
                input_maltra,input_abusex):
    newEntry = {
            'Edad_Reg': [input_edad],
            'Genero': [input_genero],
            'depresion': [input_depre],
            'Ansiedad': [input_ansie],
            'TrasPersonalidad': [input_person], 
            'TrasBipolar': [input_bipo],
            'ProblIdentSexual': [input_idensex],
            'IdeacionSuicida': [input_suici], 
            'Alcohol': [input_alcoh],
            'TMCSustanciasPsicoa': [input_sust],
            'TrasEsquizo': [input_trasesq],
            'Esquizofrenia': [input_esquiz], 
            'Obesidad': [input_obesi], 
            'DolorCronico': [input_dolor],
            'Maltrato': [input_maltra],
            'AbusoSexual': [input_abusex] 
            }
  
    newEntry = pd.DataFrame(newEntry)
    #newEntry["Suicide"] = encoder.transform(newEntry["Suicide"])
    prediccion = modelo.predict(newEntry)[0]
  
    return 'El paciente '+("SI" if prediccion == 1 else "NO")+' presenta riesgo de suicidio inminente.'
  
# Lo que ejecutara con el comando "python RunApp.py"
if __name__ == '__main__':
    app.run(debug=False) # Se inicia la aplicacion en modo debug