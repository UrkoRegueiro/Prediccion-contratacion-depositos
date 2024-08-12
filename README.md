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


<details close>
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

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En este subapartado se realizan los siguientes cambios:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Se descarta la columna `ID` por no ser relevante para el estudio.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Se convierte la columna `fecha_contacto` a `datetime`.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Se convierten las columnas booleanas a `int`.<br>
  
</details>

<details close>  
  <summary>
    <h2><ins><strong> 4. Análisis exploratorio de datos </strong></ins></h2>
  </summary>
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.1.<ins><strong> Valores duplicados </strong></ins></h3>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No se encuentran valores duplicados.
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.2.<ins><strong> Porcentaje de valores nulos por variable </strong></ins></h3>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se presentan tres variables con valores nulos:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `educacion`: 4.10% de valores nulos.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `tipo_contacto`: 28.80% de valores nulos.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `resultado_campanas_anteriores`: 81.70% de valores nulos.<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Esto será relevante a la hora de procesar los datos para el modelo predictivo. Se realizará en secciones posteriores el tratamiento<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pertinente.
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.3.<ins><strong> Balance de clases </strong></ins></h3>

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/bal_clases.png)

  </div>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Podemos observar como el `88% de los clientes que han participado` en la campaña telefónica `no han contratado un deposito` a plazo<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fijo.
  
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.4.<ins><strong> Exploración de variables categoricas </strong></ins></h3>
  
  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/val_cat.png)

  </div>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa que:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· La `mayor parte de los clientes` contactados son trabajadores: `blue-collar, management y technician`, siendo los management     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; el que presenta mayor porcentaje de contratación entre ellos con un 24.50%.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· En cuanto al estado civil `predominan` los `clientes casados`, siendo estos los que más contratan en su grupo, con un 52%.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `nivel de estudios mayoritario` es `secundaria/superiores` siendo el 49% de ellos contratantes.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· De los clientes contactados solo el 18% contrató un deposito en la campaña anterior. Observamos como se han perdido clientes     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; que contrataron en la campaña anterior pero se han ganado clientes que no habían contratado o estaban indecisos en la anterior.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· La mayor parte de los contactos han sido por móvil.
  
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.5.<ins><strong> Exploración de variables numéricas </strong></ins></h3>

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/val_num.png)

  </div>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En un primer examen se observa que:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `1.80%` de los clientes contactados `tiene deuda`.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `55.60%` de los clientes es `propietario de al menos un inmueble`.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `16%` de los clientes `tiene contratado un prestamo`.<br>
  <br>

  ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<ins>Investiguemos más en detalle cada una de las variables</ins>:<br>

  ---
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Edad`

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_edad.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa que la `edad` de la `mayor parte` de los clientes se encuentra `entre los 25 y los 60 años`.<br>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Podemos apreciar en el boxplot del centro como hay una `menor edad` en clientes que `contratan` con respecto a los que no.

  ---

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Saldo`

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_saldo.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se aprecia como los clientes que contratan depósitos parecen mantener mayor saldo en su cuenta con respecto a los que no lo hacen.
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Esto lo observamos tanto en el boxplot central como en la distribución, presentando esta un leve desplazamiento hacia salarios altos.



  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa una gran cantidad de clientes sin saldo en su cuenta. Veamos si presentan claras diferencias con los clientes con saldo a la
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hora de contratar un depósito:<br>

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/contratacion_saldo.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Podemos observar como los `clientes sin saldo contratan menos` que los que tienen.

  ---

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Duración`

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Aqui exploraremos la duración en segundos de las últimas llamadas (donde se gana o pierde un cliente).

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_duracion.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa con claridad como los `clientes que contratan` han pasado `más tiempo en la última llamada`.

  ---
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Campaña`

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Veamos como se distribuye el número de veces que se ha contactado con cada cliente en esta campaña:

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_campaña.png)

  </div>

  ---
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Tiempo transcurrido`

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Aquí exploramos la distribución de los días que han transcurrido desde la última llamada, excluyendo los que no fueron contactados             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;previamente:

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_tiempo.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se observa que para los clientes que no acaban contratando los depositos transcurren una mayor cantidad de días tras el último
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;contacto, no    así para los que contratan que son contactados con más prontitud.

  ---
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Contactos anteriores`

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Notar que no se han tenido en cuenta los clientes que no fueron contactados en campañas anteriores.

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/distr_contactos.png)

  </div>

  ---
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• `Deuda, Vivienda y Prestamo`

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/var_bool.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Se infiere de los gráficos que:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `1.80%` de los clientes contactados `tiene deuda`.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `55.60%` de los clientes es `propietario de al menos un inmueble`.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `16%` de los clientes `tiene contratado un prestamo`.<br>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;De las cifras se descubre que hay un perfil de clientes con mayor contratación:<br>
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Clientes `sin deuda`, `sin inmuebles` y `sin prestamos`.

  ---

  ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<ins>Se realiza a su vez un análisis de dispersión para comprobar si se presentan grupos diferenciados</ins>:<br>

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/dispersion.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Del gráfico inferimos que:<br>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + `Saldo`<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Parece observarse una mayor contratación en saldos más bajos, esto puede deberse a que mayores     saldos alojan capital en
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;depositos/activos con mayor riesgo y remuneración por poseer mayor sofisticación financiera(Referencia).<br>
  <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + `Edad (joven<=25 || 26<adulto<59 || mayor>=60)`<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Parece observarse una mayor contratación en personas mayores con saldo bajo (grupo recurrente).<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Se observa un menor tiempo de llamada y contacto en personas mayores y jovenes.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Se observa una mayor cantidad de llamadas a adultos para tratar de ganar contrataciones (grupo de interés).<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Parece que jovenes y mayores contratan más rapidamente tras contacto que adultos.

  
</details>

<details open>  
  <summary>
    <h2><ins><strong> 5. Análisis exploratorio de datos por grupo </strong></ins></h2>
  </summary>

  En esta sección analizaremos en detalle que tipo de clientes contratan los depositos a plazo fijo. Esto nos permitirá hacernos una idea del perfil de estos clientes para seleccionarlos en futuras campañas y ahorrar       costes.
  
  Para ello vamos a centrar el análisis en los grupo de edad:
  -  Estudiaremos que grupo proporciona mayores ingresos y la tasa de éxito de cada uno.
  -  Analizaremos cada grupo por separado, identificando los diferentes perfiles.
  -  Realizaremos un análisis de clustering para determinar si encontramos similitudes con el resultado del punto anterior, ganando confianza en los grupos de interes identificados.

  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.1.<ins><strong> Preparación del dataset </strong></ins></h3>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vamos a visualizar el histograma del número de contactos y clientes para seleccionar los intervalos de edad:

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/int_edad.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Observamos como se dirige un esfuerzo en captar clientes en edades comprendidas entre 25 y 60 años estableciendo un mayor número
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;de contactos para tratar de conseguir contrataciones. Esto es lógico ya que la mayor parte de la población española activa está en este
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grupo y por tanto ha de intentarse máximizar el número de clientes en este grupo más extenso y diverso.

  <div align="center">

  <img src="https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/grupos_edad.png" alt="Instituto Nacional de Estadística" width="700">

  <p><em>Fuente: Instituto Nacional de Estadística</em></p>

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Estudiaremos en los apartados siguientes las carácteristicas en cada grupo, para ello se establecen los siguientes criterios de
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;diferenciación de grupos:<br>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `Jovenes`: edades menores o iguales a 25 años.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `Adultos`: edades comprendidas entre 26 y 59 años.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· `Mayores`: edades mayores o iguales a 60 años.<br>

  
  <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.2.<ins><strong> Análisis de la tasa de éxito </strong></ins></h3>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Comenzaremos por visualizar la cantidad de clientes que tenemos en cada grupo:

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/clientes_grupo.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;La mayor parte de los clientes en esta campaña pertenece al grupo `adulto` seguido del grupo `joven` y finalmente `mayor`.

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Veamos ahora el porcentaje de éxito total(izquierda) y proporcional(derecha) en estos grupos:

  <div align="center">

  ![](https://github.com/UrkoRegueiro/Prediccion-contratacion-depositos/blob/master/utiles/imagenes/porcentaje_exito.png)

  </div>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Definimos la tasa de exito total como el porcentaje de éxito que ha tenido cada grupo en la campaña. Del gráfico de la izquierda anterior
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;observamos que:

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El `grupo de adultos` presenta la `mayor cantidad de contrataciones` con un 9.66%, esto se debe a que son el grupo mayoritario.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· El grupo de jovenes y mayores sigue al anterior con una tasa de exito del 0.71% y 1.34% respectivamente.<br>

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A pesar de que el `grupo de adultos` genere los mayores ingresos, dentro de su propio grupo es el que presenta `la menor tasa de`
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`éxito`, siendo esta de un `10%`. En   cuanto al resto de grupos encontramos en los `jovenes` una tasa de éxito del `24%` y en el de `mayores`
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;del `34%`.

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Para hacernos una idea de los perfiles contratantes en cada grupo realizaremos un análisis de cada uno de ellos.

  
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
