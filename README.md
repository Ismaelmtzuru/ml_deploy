# Taxi Fare Prediction App
[Script de app(streamlit)](app.py)

Este repositorio contiene una aplicación web desarrollada con Streamlit para predecir el costo de viajes en taxi en la ciudad de Nueva York (NYC). Se utiliza el conjunto de datos de taxi y el modelo de regresión creado con PyCaret.

## Configuración del entorno

Asegúrate de tener las siguientes bibliotecas instaladas antes de ejecutar la aplicación:

```bash
pip install pycaret streamlit pandas numpy Pillow
```

## Cargar el Modelo

El modelo de regresión para la predicción del costo de viaje se carga utilizando PyCaret. 

```python
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

modelo = load_model('deployment1')
```

## Funciones de Predicción

Se definen las funciones necesarias para realizar predicciones en tiempo real o en lote.

```python
def predict(model, df_input):
    prediccion_df = predict_model(estimator=model, data=df_input)
    predicciones = prediccion_df.iloc[0]['prediction_label']
    return predicciones

def run():
    # ... Código para la interfaz de usuario Streamlit ...
```

## Interfaz de Usuario

La interfaz de usuario de la aplicación permite al usuario seleccionar entre predicciones en línea o por lotes. Además, se proporcionan opciones para ingresar datos relevantes y realizar predicciones.

```python
if __name__ == '__main__':
    run()
```

## Configuración y Entrenamiento del Modelo

Se presenta el código para cargar el conjunto de datos, configurar el entorno de PyCaret y entrenar el modelo de regresión (Logistic Regression).

```python
import pandas as pd
from pycaret.regression import *

# Cargar el conjunto de datos
df = pd.read_parquet('dataset\\verde_final.parquet')

# Configuración de PyCaret
s = setup(df, target='total_amount', session_id=123)

# Crear el modelo Logistic Regression
lr = create_model('lr')
```

## Resultados del Modelo

Se muestran los resultados del modelo de regresión, incluidas las métricas de rendimiento y visualizaciones importantes.

```python
# Resultados del modelo
plot_model(lr, plot='feature')
plot_model(lr, plot='error')
plot_model(lr, plot='residuals')
```

## Despliegue de la Aplicación

El modelo entrenado se guarda para su uso en la aplicación web con Streamlit.

```python
# Guardar el modelo para el despliegue en Streamlit
save_model(lr, model_name='deployment1')
```

## Ejecutar la Aplicación

Para ejecutar la aplicación, utiliza el siguiente comando en la terminal:

```bash
streamlit run nombre_del_script.py
```

¡Disfruta de la predicción del costo de viaje en taxi en NYC! <br>
[Prediccón de costo de viaje (online)](https://ml-deploy-9c4d57010124.herokuapp.com/) <br>
[Notebook](ml_consumible.ipynb) <br>
