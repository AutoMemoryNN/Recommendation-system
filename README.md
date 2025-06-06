# Pipeline de Procesamiento de Datos

Este repositorio implementa un pipeline completo para extraer, procesar, tokenizar y guardar datasets en formato NumPy. La aplicación combina datos de diversas fuentes (por ejemplo, Coursera, edX, Udemy, etc.) y genera un archivo consolidado para su análisis posterior.

## Requisitos Previos

### Archivos de Datos Crudos

Coloca en la carpeta `raw-data` los archivos CSV necesarios para el procesamiento.  
Puedes descargarlos desde el siguiente enlace:  
[Descargar archivos CSV](https://drive.google.com/file/d/1AEtY4SA8eSnKnqY3mUdKkqlodFYsVI0U/view?usp=drive_link)

### Documentación Adicional

Para entender mejor el funcionamiento y los fundamentos teóricos de este proyecto,  
se recomienda leer el siguiente paper:  
[Leer el Paper](https://drive.google.com/file/d/1NGDWHxB3GOl4YdZb1ywar0WPKOkuXWIo/view?usp=drive_link)

### Prerequisitos

- **Python 3.8+**
- **Git** (opcional, para clonar el repositorio)

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2. Crear y Activar un Entorno Virtual (Opcional)

Recomendamos utilizar un entorno virtual para aislar las dependencias:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Crear Directorios requeridos

Se encesitan estos directorios

```bash
mkdir -p data/csv_data data/x_data data/y_data filtered-data/non-tokenized raw-data
```

### 4. Instalar las Dependencias

El proyecto depende de las siguientes librerías:
- pandas
- numpy
- tensorflow
- matplotlib
- transformers
- spacy

Si cuentas con un archivo requirements.txt, instala las dependencias con:

```bash
pip install -r requirements.txt
```

Si no dispones de este archivo, crea uno con el siguiente contenido:

```
pandas
numpy
tensorflow
matplotlib
transformers
spacy
```

### 5. Descargar el Modelo de SpaCy

El pipeline utiliza el modelo de idioma inglés de SpaCy. Descárgalo ejecutando:

```bash
python -m spacy download en_core_web_sm
```

## Estructura del Proyecto

La estructura del repositorio es la siguiente:

```
tu_repositorio/
├── dataManager/
│   ├── DataExtractor.py    # Contiene la clase DataExtractor
│   └── CsvManager.py       # Módulo de CSV y asignación de categorías
├── src/
│   ├── Main.py             # Punto de entrada de la aplicación
│   ├── FFNNImplementation.py
│   ├── Predictor.py
│   └── ...                 # Otros módulos y archivos relacionados
├── categories.json         # Archivo con mapeo de categorías para temas
├── requirements.txt
└── README.md
```

## Ejecución del Pipeline

El archivo principal se encuentra en `src/Main.py` y contiene el siguiente código:

```python
from dataManager.DataExtractor import DataExtractor

def main():
    de = DataExtractor()
    de.DEFAULT_TOP_TOPICS = 40
    de.process_pipeline()
    de.print_topic_distribution()

main()
```

La función `process_pipeline` ejecuta estas etapas:

1. **Extracción de datos**: Se extraen datasets desde diversas fuentes.
2. **Generación de datos consolidados**: Se combinan los datos extraídos en un único archivo.
3. **División y tokenización**: Se separan los datos en conjuntos de entrenamiento, validación y prueba, y se tokenizan.
4. **Guardado en formato NumPy**: Se guardan los datasets procesados en archivos .npy.

Para ejecutar el pipeline, simplemente usa:

```bash
python src/Main.py
```

## Personalización y Parámetros

La función `process_pipeline` acepta parámetros opcionales para ajustar:

- `n_top_topics`: Número de temas principales a conservar (por defecto, se establece en 40 en el main).
- `val_ratio`: Proporción de datos para el conjunto de validación.
- `test_ratio`: Proporción de datos para el conjunto de pruebas.

Estos parámetros se pueden modificar directamente en el código según tus necesidades.

## Contribuciones

Si deseas contribuir a este proyecto, por favor:

1. Abre un issue para discutir posibles mejoras.
2. Envía un pull request con tus cambios.