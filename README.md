# Análisis Comparativo de Series Temporales: Predicción de Tiempos de Vuelta en F1

Este proyecto es un estudio comparativo exhaustivo sobre el pronóstico de series temporales, implementando y evaluando modelos estadísticos clásicos (ARIMA, SARIMA) y arquitecturas de Deep Learning (RNN, LSTM) para predecir los tiempos de vuelta en la Fórmula 1.


## Descripción del Proyecto

  - Este proyecto presenta el desarrollo y la evaluación de cuatro modelos de pronóstico de series temporales para predecir el rendimiento de los pilotos, utilizando la dramática carrera del **GP de Abu Dhabi 2021** como caso de estudio.
  - Se implementan múltiples enfoques, desde un modelo ARIMA simple para una sola tanda, un modelo SARIMA para manejar los ciclos de la carrera completa, hasta redes neuronales (RNN y LSTM) apiladas con la API Funcional de Keras.
  - El objetivo es realizar un análisis comparativo riguroso, no solo entre las arquitecturas de modelo, sino también entre el enfoque estadístico clásico y el enfoque de Deep Learning, para determinar la solución más efectiva para este problema de pronóstico.

## Análisis y Metodología

### El proyecto se estructura en las siguientes fases:

1.  **Análisis Exploratorio de Datos (EDA) y Preprocesamiento:**

      * Se utilizó la librería `fastf1` para cargar y procesar los datos de telemetría y vueltas de la carrera.
      * Se analizaron visualmente las series temporales de los pilotos (Verstappen y Hamilton) para identificar **tendencias** (causadas por la degradación de neumáticos) y **estacionalidad** (patrones cíclicos creados por las paradas en boxes).
      * Se aplicó la limpieza de datos para eliminar *outliers* claros (ej. vueltas bajo Safety Car o de formación) y se convirtieron los tiempos de vuelta a segundos.
      * Se generaron gráficos **ACF y PACF** para informar la selección de parámetros de los modelos estadísticos.

2.  **Modelo 1 y 2: Modelos Estadísticos Clásicos (ARIMA/SARIMA)**

      * **Técnica:** Se utilizaron los modelos de la familia ARIMA para capturar la autocorrelación en los datos.
      * **ARIMA (Fase 1):** Se ajustó un modelo ARIMA simple `(p,d,q)` a una sola tanda de carrera (*stint*) para modelar la tendencia de degradación lineal.
      * **SARIMA (Fase 2):** Se ajustó un modelo SARIMA `(p,d,q)(P,D,Q,m)` a la carrera completa. El componente estacional se utilizó para aprender el patrón cíclico de las paradas en boxes (ej. `m=22`, la duración de una tanda).
      * **Resultado:** Dos modelos estadísticos que sirven como una sólida línea base de rendimiento.

3.  **Modelo 3 y 4: Enfoque de Deep Learning (RNN/LSTM)**

      * **Técnica:** Se construyeron dos arquitecturas de redes neuronales recurrentes utilizando la **API Funcional de Keras** para una comparación directa.
      * **`SimpleRNN`:** Se implementó una red RNN apilada (`32 -> 16 -> 8`).
      * **`LSTM`:** Se implementó una red LSTM apilada con la misma arquitectura para evaluar el impacto de la "memoria a largo plazo" en el rendimiento.
      * **Arquitectura:** Ambos modelos incluyeron capas de `LayerNormalization` y `Dropout` para estabilizar el entrenamiento y prevenir el sobreajuste.
      * **Optimización:** Se utilizó **Keras Tuner** (con `Hyperband`) para realizar una búsqueda sistemática de los mejores hiperparámetros (unidades de capa, tasa de dropout y tasa de aprendizaje) para el modelo más prometedor.

4.  **Comparación Final de Modelos**

      * Se consolidaron las métricas de rendimiento clave (**RMSE** y **MAE**) de los cuatro modelos finales en una tabla comparativa.
      * Se generaron gráficos para visualizar el ajuste de cada modelo frente a los datos reales de la carrera, discutiendo cómo cada enfoque manejó los patrones de degradación y los eventos aleatorios (como el Safety Car al final de la carrera).


## Tecnologías Utilizadas

  - **Python 3.10+**
  - **Pandas & NumPy:** Para manipulación y preparación de datos.
  - **Matplotlib & Seaborn:** Para visualización de datos y correlogramas.
  - **Scikit-learn:** Para métricas de evaluación (RMSE, MAE) y `MinMaxScaler`.
  - **fastf1:** Para la ingesta y procesamiento de datos de carreras de Fórmula 1.
  - **statsmodels:** Para la implementación de los modelos ARIMA y SARIMA.
  - **pmdarima:** Para la función `auto_arima` y la selección de parámetros.
  - **TensorFlow & Keras:** Para la construcción, entrenamiento y optimización de los modelos RNN y LSTM.
  - **Keras Tuner:** Para la optimización automática de hiperparámetros.
  - **Google Colab (o Jupyter Notebook):** Como entorno de desarrollo.


## Instrucciones de Instalación

1.  **Clona este repositorio**

    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio-f1-series-temporales
    cd tu-repositorio-f1-series-temporales
    ```

2.  **Crea un entorno virtual (recomendado)**

    ```bash
    python -m venv f1_env
    source f1_env/bin/activate  # En Windows: f1_env\Scripts\activate
    ```

3.  **Instala las dependencias**

    Crea un archivo `requirements.txt` con el siguiente contenido y luego ejecuta `pip install -r requirements.txt`.

    ```txt
    # Frameworks de DL y Optimización
    tensorflow
    keras-tuner

    # Frameworks de Series Temporales
    statsmodels
    pmdarima

    # Ingesta de Datos F1
    fastf1

    # Análisis y Métricas
    pandas
    numpy
    scikit-learn

    # Visualización
    matplotlib
    seaborn

    # Entorno de Notebook
    notebook
    ```

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta los notebooks**

    Inicia Jupyter Notebook en el entorno activado y abre los archivos `.ipynb`:

    ```bash
    jupyter notebook
    ```
