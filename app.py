from pycaret.regression import load_model,predict_model
import streamlit as st 
import pandas as pd 
import numpy as np  
from PIL import Image


modelo = load_model('deployment1')

def predict(model,df_input):
    prediccion_df = predict_model(estimator=model, data=df_input)
    

    predicciones = prediccion_df.iloc[0]['prediction_label']
    return predicciones 

def run():
    image_hospital = Image.open('img\\hospital1.jpg')

    add_selectbox= st.sidebar.selectbox(
        'Como te gustaría predecir', ('Online','Batch')
    )

    st.sidebar.info('Esta app es creada para predecir el costo de viaje de taxis en NYC')
    st.sidebar.success('https://www.pycaret.org')

    st.sidebar.image(image_hospital)

    #Titulo de la aplicación
    st.title('App de Predicción de costo de viaje en taxi')

    #Selecciones al seleccionar la forma de prediccion online

    if add_selectbox == 'Online':
        tipo = st.selectbox('Tipo(1=En calle,2=Llamada)',[1,2])
        distancia = st.number_input('Distancia(millas)',min_value=1,max_value=300,value=10)
        pasajero = st.number_input('Total de pasajeros',min_value=1,max_value=4,value=1)
        calculada = st.number_input('Valor estimado',min_value=3.00,max_value=500.00,value=5.00)

        dict_input={'trip_type':tipo,'trip_distance':distancia,'passenger_count':pasajero,'fare_amount':calculada}
        df_input=pd.DataFrame([dict_input])

        if st.button('Predict'):
            output = predict(model=modelo,df_input=df_input)
            output = '$' + str(output)

            st.success('Costo aproximado de viaje(USDollar): {}'.format(output))

    # Para seleccion Batch
    if add_selectbox == 'Batch':
        file_upload=st.file_uploader('Carga el archivo .parquet para las predicciones',type=['parquet'])

        if file_upload is not None:
            data=pd.read_parquet(file_upload)
            predictions = predict_model(estimator=modelo,data=data)
            st.write(predictions)

# Se ejecuta la aplicaci´no
if __name__ == '__main__':
    run()