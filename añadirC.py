import pandas as pd

# Cargar el archivo train_dataset.csv
train_df = pd.read_csv('train_dataset.csv')

# Agregar la nueva columna "problem_category"
train_df['problem_category'] = ''

# Guardar el archivo modificado
train_df.to_csv('train_dataset.csv', index=False)

# Cargar el archivo validation_dataset.csv 
val_df = pd.read_csv('validation_dataset.csv')

# Agregar la nueva columna "problem_category"
val_df['problem_category'] = ''

# Guardar el archivo modificado
val_df.to_csv('validation_dataset.csv', index=False)