import pandas as pd

from functions import inicializar, consulta, cleaner_pipeline, prediccion
from dotenv import load_dotenv
load_dotenv()

# Busqueda en bases de datos:

MYSQL_PASS, query = inicializar()

resultado = consulta(MYSQL_PASS, query)

clientes, datos_filtrados = cleaner_pipeline(resultado)

clientes_potenciales = prediccion(clientes, datos_filtrados)

print(clientes_potenciales)
