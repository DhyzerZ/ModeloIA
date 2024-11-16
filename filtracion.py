import pandas as pd
import numpy as np
from sklearn.utils import resample
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def enhanced_clean_text(text):
    """
    Limpieza mejorada del texto con lemmatización y manejo de caracteres especiales
    """
    # Convertir a minúsculas
    text = str(text).lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Reemplazar números por token especial
    text = re.sub(r'\d+', 'NUM', text)
    
    # Eliminar caracteres especiales pero mantener puntuación básica
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Lemmatización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Unir tokens
    return ' '.join(tokens).strip()

def filter_english_reviews(df):
    """
    Filtro mejorado para reseñas en inglés
    """
    stop_words = set(stopwords.words('english'))
    
    def is_english(text):
        words = set(str(text).lower().split()) 
        english_word_count = len(words.intersection(stop_words))
        return english_word_count >= 3
    
    return df[df['cleaned_text'].apply(is_english)]

def verify_and_clean_final(df):
    """
    Verificación final y limpieza adicional
    """
    # Verificar longitud después de limpieza
    df['final_length'] = df['cleaned_text'].str.len()
    df = df[df['final_length'] >= 10]  # Asegurar longitud mínima después de limpieza
    
    # Verificar caracteres válidos
    valid_chars_pattern = re.compile(r'^[a-zA-Z0-9\s.,!?]+$')
    df = df[df['cleaned_text'].apply(lambda x: bool(valid_chars_pattern.match(str(x))))]
    
    return df

def create_balanced_sample(df, sample_size_per_class):
    """
    Crear muestra balanceada del dataset
    """
    df_positive = df[df['review_score'] == 1]
    df_negative = df[df['review_score'] == -1]
    
    sample_size = min(sample_size_per_class, len(df_positive), len(df_negative))
    
    df_pos_sampled = resample(df_positive, n_samples=sample_size, random_state=42)
    df_neg_sampled = resample(df_negative, n_samples=sample_size, random_state=42)
    
    return pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42)

def final_clean_dataset(file_path, train_size=0.03, validation_size=0.10, min_length=20, max_length=5000):
    """
    Versión final de limpieza y preparación del dataset
    """
    print("Cargando y procesando dataset...")
    df = pd.read_csv(file_path)
    initial_size = len(df)
    
    # 1. Eliminar filas con valores nulos
    df = df.dropna(subset=['review_text'])
    df['app_name'] = df['app_name'].fillna('Unknown Game')
    print(f"Registros después de eliminar nulos: {len(df)} ({len(df)/initial_size*100:.2f}%)")
    
    # 2. Filtrar por longitud de reseña
    df['review_length'] = df['review_text'].str.len()
    df = df[(df['review_length'] >= min_length) & (df['review_length'] <= max_length)]
    print(f"Registros después de filtrar por longitud: {len(df)} ({len(df)/initial_size*100:.2f}%)")
    
    # 3. Limpieza mejorada del texto
    print("Aplicando limpieza mejorada del texto...")
    df['cleaned_text'] = df['review_text'].apply(enhanced_clean_text)
    
    # 4. Detectar y filtrar idioma
    print("Filtrando por idioma...")
    df = filter_english_reviews(df)
    print(f"Registros después de filtrar por idioma: {len(df)} ({len(df)/initial_size*100:.2f}%)")
    
    # 5. Eliminar duplicados usando el texto limpio
    df = df.drop_duplicates(subset=['cleaned_text'])
    print(f"Registros después de eliminar duplicados: {len(df)} ({len(df)/initial_size*100:.2f}%)")
    
    # 6. Verificación final de calidad
    df = verify_and_clean_final(df)
    
    # 7. Calcular tamaños de conjunto
    total_desired = int(initial_size * (train_size + validation_size))
    train_ratio = train_size / (train_size + validation_size)
    train_size = int(total_desired * train_ratio)
    validation_size = total_desired - train_size
    
    # 8. Crear muestras balanceadas
    df_train = create_balanced_sample(df, sample_size_per_class=train_size//2)
    df_validation = create_balanced_sample(
        df[~df.index.isin(df_train.index)], 
        sample_size_per_class=validation_size//2
    )
    
    print(f"\nTamaño final conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño final conjunto de validación: {len(df_validation)}")
    
    return df_train, df_validation

def save_final_datasets(df_train, df_validation, train_path='train_dataset.csv', val_path='validation_dataset.csv'):
    """
    Guardar datasets procesados
    """
    columns_to_save = ['app_id', 'app_name', 'review_text', 'review_score', 'cleaned_text', 'review_length']
    df_train[columns_to_save].to_csv(train_path, index=False)
    df_validation[columns_to_save].to_csv(val_path, index=False)
    print(f"\nDatasets finales guardados en: {train_path} y {val_path}")

def main():
    # Configuración
    input_file = 'dataset.csv'
    train_size = 0.03
    validation_size = 0.10
    
    # Procesar datasets
    df_train, df_validation = final_clean_dataset(
        input_file, 
        train_size=train_size, 
        validation_size=validation_size
    )
    
    # Guardar datasets procesados
    save_final_datasets(df_train, df_validation)
    
    print("\nProcesamiento completado. Los datasets están listos para el modelado.")
    
    return df_train, df_validation

if __name__ == "__main__":
    df_train, df_validation = main()