<h1 align="center"> Estudio y predicción en la contratación de depositos bancarios. </h1>

<div align="center">

  <img src="https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/portada.png" alt="Caso de uso" width="80%">
  
</div>

## Tecnologías usadas

**Lenguaje:** Python.

**Librerias:** numpy, pandas, matplotlib, seaborn, plotly, sklearn, xgboost, imblearn, regex

------------

<h2>
  
Para visualizar la versión detallada del presente proyecto véase: <br>

<div align="center">

  [Estudio y predicción en la contratación de depositos bancarios.](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/caso_uso.ipynb)
  
</div>

</h2>
  
------------


<details open>
  <summary>
    <h2><ins><strong> 1. Introducción </strong></ins></h2>
  </summary>
  
  En el ámbito del contexto bancario, un depósito a plazo fijo es un producto ofrecido por el banco y es una de sus fuentes principales de ingresos. Un depósito a plazo fijo es un producto financiero de ahorro donde 
  el cliente deposita una cantidad fija de dinero en una entidad bancaria durante un periodo de tiempo establecido. Este tipo de depósitos ofrecen una rentabilidad traducida en intereses para el cliente y a su vez es
  dinero activo para la entidad bancaria.
  
  Desde una entidad bancaria anónima, han lanzado diferentes estrategias para atraer a sus clientes a contratar este tipo de depósito. Las campañas telefónicas son las que más éxito siguen teniendo. Sin embargo, para
  llevar a cabo una campaña telefónica se requiere un gran esfuerzo debido al gran número de clientes que puede poseer una entidad bancaria. Así, uno de los <u>**objetivos**</u> que se ha propuesto la entidad bancaria es   <u>**identificar
  aquellos clientes que puedan tener más propensión a la contratación**</u> del producto y realizar únicamente las llamadas a este tipo de clientes.

  El conjunto de datos está relacionado con una campaña telefónica que realizó una entidad bancaria.

  <u>**OBJETIVOS**</u>:

  - Realizar un <u>**análisis exploratorio**</u> sobre los diferentes <u>**tipos de clientes**</u>.

  - <u>**Predecir que clientes**</u> del conjunto de datos <u>**contratarán**</u> o no un depósito.

  - <u>**Explicación**</u> de los <u>**resultados**</u> obtenidos.

  - ¿Cómo llevarías la solución desarrollada a producción y qué consideraciones tendrías en cuenta?
  
</details>

<details close>
  <summary>
    <h2><ins><strong> 2. Importación de paquetes y Set de datos </strong></ins></h2>
  </summary>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.1.<ins><strong> Paquetes </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.2.<ins><strong> DataSet </strong></ins></h3>
  
</details>

<details close>
  <summary>
    <h2><ins><strong> 3. Inspección general y preprocesamiento de datos </strong></ins></h2>
  </summary>

  En esta sección se realizarán las transformaciones necesarias en los datos para realizar un análisis general de estos y tener un primer acercamiento sobre las variables categóricas y numéricas, extrayendo insights        generales de nuestros clientes.

  | Column                        | Description                                                                                 | Type   |
  |-------------------------------|---------------------------------------------------------------------------------------------|--------|
  | ID                            | Identificador del cliente.                                                                  | Int    |
  | trabajo                       | Variable categórica donde indica a qué se dedica un cliente.                                | Object |
  | edad                          | Edad del cliente                                                                            | Int    |
  | estado_civil                  | Estado civil del cliente: casado, divorciado, soltero                                       | Object |
  | educacion                     | Nivel de estudios del cliente: primaria, secundaria/superior, universitarios                | Object |
  | deuda                         | Variable booleana que indica si el cliente tiene alguna deuda pendiente: sí, no             | Object |
  | saldo                         | Saldo que tiene el cliente en la cuenta.                                                    | Int    |
  | vivienda                      | Variable booleana que indica si el cliente tiene una vivienda en propiedad: sí, no          | Object |
  | prestamo                      | Variable booleana que indica si el cliente tiene un préstamo: sí, no                        | Object |
  | tipo_contacto                 | Indica cómo se ha realizado el contacto con el cliente: movil, teléfono                     | Object |
  | duracion                      | Indica la duración en segundos de la última llamada.                                        | Int    |
  | fecha_contacto                | Indica la última fecha de contacto con el cliente.                                          | Object |
  | campaign                      | Indica el número de veces que se ha contactado con el cliente para la campaña actual.       | Int    |
  | tiempo_transcurrido           | Número de días que han transcurrido desde la última llamada. -1 (cliente no fue contactado previamente) | Int    |
  | contactos_anteriores          | Indica el número de veces que se ha contactado con el cliente para campañas anteriores.     | Int    |
  | resultado_campanas_anteriores | Indica el resultado obtenido en campañas anteriores: éxito, sin_éxito, otro                 | Object |
  | target                        | Variable booleana que indica si el cliente ha contratado el producto para la actual campaña: 0-NO, 1-SÍ | Int    |

  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.1.<ins><strong> Procesamiento inicial de datos </strong></ins></h3>
  En este subapartado se realizan los siguientes cambios:

- Se descarta la columna `ID` por no ser relevante para el estudio.
- Se convierte la columna `fecha_contacto` a `datetime`.
- Se convierten las columnas booleanas a `int`.
  
</details>

<details close>  
  <summary>
    <h2><ins><strong> 4. Análisis exploratorio de datos </strong></ins></h2>
  </summary>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.1.<ins><strong> Valores duplicados </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.2.<ins><strong> Porcentaje de valores nulos por variable </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.3.<ins><strong> Balance de clases </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.4.<ins><strong> Exploración de variables categoricas </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.5.<ins><strong> Exploración de variables numéricas </strong></ins></h3>
</details>

<details close>  
  <summary>
    <h2><ins><strong> 5. Análisis exploratorio de datos por grupo </strong></ins></h2>
  </summary>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.1.<ins><strong> Preparación del dataset </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.2.<ins><strong> Análisis de la tasa de éxito </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.3.<ins><strong> Análisis de los perfiles por grupo </strong></ins></h3>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.3.1.<ins><strong> Distribución de saldos por grupo </strong></ins></h4>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.3.2.<ins><strong> Adultos </strong></ins></h4>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.3.3.<ins><strong> Jovenes </strong></ins></h4>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.3.4.<ins><strong> Mayores </strong></ins></h4>
</details>

<details close>  
  <summary>
    <h2><ins><strong> 6. Modelaje </strong></ins></h2>
  </summary>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.1.<ins><strong> Tratamiento de valores nulos </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.2.<ins><strong> Tratamiento de outliers </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.3.<ins><strong> Transformación de columnas </strong></ins></h3>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.3.1.<ins><strong> Limpieza columnas </strong></ins></h4>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.3.2.<ins><strong> Encoding </strong></ins></h4>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.3.3.<ins><strong> Eliminación de columnas </strong></ins></h4>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.4.<ins><strong> Balance de clases </strong></ins></h3>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.5.<ins><strong> Selección de Modelo </strong></ins></h3>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.5.1.<ins><strong> Cross Validation </strong></ins></h4>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.6.<ins><strong> Tuning GradientBoostingClassifier </strong></ins></h3>
  <h4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.6.1.<ins><strong> Feature importance </strong></ins></h4>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6.7.<ins><strong> Resultados y Conclusiones </strong></ins></h3>
</details>


<h1 align="center"></h1>
