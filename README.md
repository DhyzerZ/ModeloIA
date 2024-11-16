# Análisis de Sentimientos en Reseñas de Juegos de Steam

### Integrantes del equipo
   #### Salomón Vélez Pérez
   #### Cristian Camilo Zapata Garcia
   #### Juan Sebastian Jacome Burbano
#


## Descripción General
Este proyecto implementa un modelo de aprendizaje automático para analizar reseñas de juegos de Steam, clasificándolas como positivas o negativas mientras, de manera secundaria, identifica tipos específicos de problemas mencionados en las reseñas. El modelo utiliza técnicas de procesamiento de lenguaje natural y un clasificador de regresión logística para proporcionar análisis de sentimientos y categorización de problemas.

## Características
- Análisis de sentimientos (clasificación positiva/negativa)
- Detección de categorías de problemas:
  - Problemas técnicos
  - Problemas de diseño
  - Problemas de monetización
  - Problemas relacionados con el contenido
- Filtrado multilingüe (enfocado en reseñas en inglés)
- Preprocesamiento y limpieza de texto
- Creación de conjunto de datos balanceado
- Métricas de evaluación completas

## Conjunto de Datos
- Fuente: [Steam Reviews Dataset](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)
- Tamaño inicial: 6,417,106 reseñas
- Después del procesamiento:
  - Conjunto de entrenamiento: 192,512 reseñas (3% del original)
  - Conjunto de validación: 641,710 reseñas (10% del original)

## Requisitos
```
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
joblib
warnings
collections
re
```

## Estructura del Proyecto
```
├── filtracion.py        # Preprocesamiento y filtrado de datos
├── analyze_datasets.py  # Análisis y visualización de datasets
├── añadirC.py           # Añade una columna a los csv para la identificacion de problemas
├── modeloP03.py         # Entrenamiento y evaluación del modelo
├── pruebaMG01.py        # Interfaz interactiva de predicción
└── README.md
```

## Pipeline de Procesamiento de Datos

### 1. Filtrado de Datos (`filtracion.py`)
```python
# Pasos clave del procesamiento
1. Limpieza de texto
   - Conversión a minúsculas
   - Eliminación de URLs
   - Estandarización de números
   - Filtrado de caracteres especiales
   - Normalización de espacios
   - Tokenización
   - Lematización

2. Filtrado por idioma
   - Solo reseñas en inglés

3. Filtrado por longitud
   - Mínimo 10 palabras
   - Máximo 5000 caracteres

4. Balanceo de datos
   - Igual número de reseñas positivas/negativas
```

Resultados del Procesamiento:
```
Registros iniciales: 6,417,106
Después de eliminar nulos: 6,409,801 (99.89%)
Después del filtrado por longitud: 6,107,918 (95.18%)
Después del filtrado por idioma: 4,459,728 (69.50%)
Después de eliminar duplicados: 3,836,867 (59.79%)
```

### 2. Análisis de Datasets (`analyze_datasets.py`)
El análisis incluye:
- Verificación de distribución de clases
- Estadísticas de longitud de texto
- Análisis de vocabulario
- Análisis de distribución de juegos
- Controles de calidad

Hallazgos Principales:
```
Distribución de Clases:
- Entrenamiento: 50% positivas, 50% negativas
- Validación: 50% positivas, 50% negativas

Estadísticas de Longitud de Texto:
Entrenamiento:
- Media: 446.51 caracteres
- Mediana: 221 caracteres
- Máximo: 4,999 caracteres

Validación:
- Media: 442.98 caracteres
- Mediana: 219 caracteres
- Máximo: 5,000 caracteres
```

Juegos con Más Reseñas:
```
1. PAYDAY 2
2. Terraria
3. Dota 2
4. Undertale
5. Grand Theft Auto V
```
### 3. Entrenamiento del Modelo (`añadirC.py`)
Este codigo crea una columna adicional en los datasets "train_dataset" y "validation_dataset" para incluir la identificacion de problemas mediante la libreria de pandas

### 4. Entrenamiento del Modelo (`modeloP03.py`)
Configuración del Modelo:
- Vectorizador TF-IDF (max_features=5000)
- Regresión Logística (max_iter=1000)

Métricas de Rendimiento:
```
Precisión: 0.85
Recall: 0.85
F1-Score: 0.85
Exactitud: 0.85
```

Distribución de Detección de Problemas:
```
Problemas de Diseño: 158,470 reseñas
Problemas de Contenido: 110,828 reseñas
Problemas Técnicos: 75,323 reseñas
Problemas de Monetización: 53,096 reseñas
```

### 5. Interfaz Interactiva (`pruebaMG01.py`)
Características:
- Análisis de reseñas en tiempo real
- Predicción de sentimientos con niveles de confianza
- Detección de categorías de problemas
- Estimaciones de probabilidad para cada tipo de problema

Ejemplo de Uso:
```
=== Predictor de Sentimientos para Reseñas de Juegos ===
> Ingresa una reseña: is a good game

Resultados:
- Sentimiento: Positivo
- Confianza: 94.33%
- Probabilidades de Sentimiento:
  * Negativo: 5.67%
  * Positivo: 94.33%
```

## Instalación y Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/DhyzerZ/ModeloIA.git
cd ModeloIA
```
2. Descargar el dataset de kaggle y ponerlo en la misma carpeta del proyecto
```
https://www.kaggle.com/datasets/andrewmvd/steam-reviews
```
3. Instalar requerimientos:
```bash
pip install -r requirements.txt
```

4. Ejecutar el pipeline:
```bash
# Procesar y filtrar datos
python filtracion.py

# Analizar datasets
python analyze_datasets.py

# Añadir la columna para la identificacion de problemas
python añadirC.py

# Entrenar modelo
python modeloP03.py

# Ejecutar interfaz interactiva
python pruebaMG01.py
```

## Visualización del Rendimiento del Modelo

### Matriz de Confusión
```
[[272036  48819]
 [ 49205 271650]]
```
Interpretación: 
- Verdaderos Negativos (272,036): - Reseñas negativas correctamente identificadas - 84.8% de las reseñas negativas
- Falsos Positivos (48,819):
   - Reseñas negativas clasificadas como positivas
   - 15.2% de error en reseñas negativas
- Falsos Negativos (49,205):
   - Reseñas positivas clasificadas como negativas
   - 15.3% de error en reseñas positivas
- Verdaderos Positivos (271,650):
   - Reseñas positivas correctamente identificadas
   - 84.7% de las reseñas positivas

### Probabilidades de Detección de Problemas
```
Problemas Técnicos: 27.04%
Problemas de Diseño: 39.88%
Problemas de Monetización: 41.65%
Problemas de Contenido: 36.91%
```

## Mejoras Futuras
1. Implementar modelos de deep learning para mejor precisión
2. Mejorar la detección de categorías de problemas
3. Implementar una mejor interfaz

## Contribución
¡Las contribuciones son bienvenidas!

## Resultados esperados de Ejecución

### Resultados de Filtración
```
Cargando y procesando dataset...
Registros después de eliminar nulos: 6409801 (99.89%)
Registros después de filtrar por longitud: 6107918 (95.18%)
Aplicando limpieza mejorada del texto...
Filtrando por idioma...
Registros después de filtrar por idioma: 4459728 (69.50%)
Registros después de eliminar duplicados: 3836867 (59.79%)
Tamaño final conjunto de entrenamiento: 192512
Tamaño final conjunto de validación: 641710
```

### Resultados del Análisis de Datasets
```
=== Cargando datasets ===
Dataset de entrenamiento: 192,512 registros
Dataset de validación: 641,710 registros
(OK) Todas las columnas requeridas están presentes

=== Distribución de Clases ===
Distribución balanceada en ambos conjuntos (50% positivas, 50% negativas)

=== Verificaciones de Calidad ===
(OK) Balance de clases
(OK) Distribución de longitudes
(OK) Calidad del vocabulario
(OK) Cobertura de juegos
(OK) Limpieza de texto
```

### Resultados del Entrenamiento del Modelo
```
Preparando datos...
Entrenando el modelo...
¡Entrenamiento completado!

Evaluando el modelo...
Informe de Clasificación de Sentimiento:
              precision    recall  f1-score   support
         -1       0.85      0.85      0.85    320855
          1       0.85      0.85      0.85    320855
   accuracy                           0.85    641710


Análisis de problemas específicos:
- Problemas de diseño: 158,470 reseñas
- Problemas de contenido: 110,828 reseñas
- Problemas técnicos: 75,323 reseñas
- Problemas de monetización: 53,096 reseñas
```
## Salidas esperadas al hacer la prueba de validacion

### Caso 1: Reseña Positiva Simple

```
Input: "is a good game"
- Sentimiento: Positivo (94.33% confianza)
- Sin problemas detectados
```
### Caso 2: Reseña Negativa con Problema Específico

```
Input: "This game is very bad, it has many bugs which makes it impossible to play, terrible experience"
- Sentimiento: Negativo (99.43% confianza)
- Problemas técnicos: 80.00% probabilidad
```
