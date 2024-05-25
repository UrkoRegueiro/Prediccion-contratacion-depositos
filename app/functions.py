# Básicos
from joblib import load
import mysql.connector
import pickle
import pandas as pd
import re
import os

def inicializar():
    pass_db = input("Introduce el nombre de la contraseña en el archivo .ENV: ")
    query = input("Introduce la consulta a la base de datos: ")

    MYSQL_PASS = os.getenv(pass_db)

    return MYSQL_PASS, query


def consulta(MYSQL_PASS, query):
    config = {'user': 'root',
              'password': MYSQL_PASS,
              'host': 'localhost',
              'database': 'clientes_banco',
              'raise_on_warnings': True
              }
    # Conexion a base de datos:
    conn = mysql.connector.connect(**config)

    # Resultados consulta:
    df_resultados = pd.read_sql(query, conn)

    # Cierro conexión
    conn.close()

    return df_resultados

def cleaner_pipeline(df):
    # Convierto variables bool a int:
    bool_dic = {"no": 0,
                "si": 1}

    df["deuda"] = df["deuda"].map(bool_dic)
    df["vivienda"] = df["vivienda"].map(bool_dic)
    df["prestamo"] = df["prestamo"].map(bool_dic)

    # Asigno categoria "desconocido" a "educacion" para tratar NaN's:
    df["educacion"] = df["educacion"].fillna("desconocida")

    # Limpio la columna educacion:
    df["educacion"] = df["educacion"].apply(lambda x: re.sub(r"\b(pri\w*)\b", "primaria", x))

    # Cargo los encoders para las columnas categóricas:
    columns = ["trabajo", "estado_civil", "educacion"]

    for num, column in enumerate(columns):
        ruta = f"utiles/encoders/{num}_{column}_encoder.pickle"

        with open(ruta, "rb") as f:
            encoder = pickle.load(f)

        data_encoded = encoder.transform(df[[column]]).toarray()
        df_encoded = pd.DataFrame(data_encoded, columns=encoder.categories_[0].tolist())
        df = pd.concat([df, df_encoded], axis=1).drop([column], axis=1)

    # Guardo los ID de los clientes para posterior contacto:
    clientes = df[["ID"]]

    # Quito las columnas que no se utilizarán:
    df = df.drop(columns=["ID", "tipo_contacto", "resultado_campanas_anteriores", "fecha_contacto"])

    return clientes, df



def prediccion(clientes, df):

    # Cargo modelo:
    model_hgbc = load("utiles/modelo/modelo_hgbc.pkl")

    # Hago predicciones:
    y_pred = model_hgbc.predict_proba(df)
    # Aplico un threshold
    threshold = 0.85
    y_pred_threshold = [0 if prediction[0] > threshold else 1 for prediction in y_pred]

    clientes_potenciales = []
    for id, contrata in zip(clientes["ID"], y_pred_threshold):
        if contrata == 1:
            clientes_potenciales.append(id)

    json_file = {"clientes_potenciales": {"ids": clientes_potenciales}}


    return json_file