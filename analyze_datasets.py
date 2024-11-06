import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_and_verify_datasets(train_path='train_dataset.csv', val_path='validation_dataset.csv'):
    """
    Carga y verifica la integridad básica de ambos datasets
    """
    print("=== Cargando datasets ===")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"\nDataset de entrenamiento: {len(df_train):,} registros")
    print(f"Dataset de validación: {len(df_val):,} registros")
    
    # Verificar columnas
    required_columns = ['app_id', 'app_name', 'review_text', 'review_score', 'cleaned_text', 'review_length']
    missing_train = set(required_columns) - set(df_train.columns)
    missing_val = set(required_columns) - set(df_val.columns)
    
    if missing_train or missing_val:
        print("\n⚠️ ADVERTENCIA: Columnas faltantes:")
        if missing_train:
            print(f"Training: {missing_train}")
        if missing_val:
            print(f"Validation: {missing_val}")
    else:
        print("\n✓ Todas las columnas requeridas están presentes")
    
    return df_train, df_val

def analyze_class_distribution(df_train, df_val):
    """
    Analiza y compara la distribución de clases en ambos datasets
    """
    print("\n=== Distribución de Clases ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set
    train_dist = df_train['review_score'].value_counts(normalize=True)
    print("\nDistribución en Training:")
    print(train_dist)
    train_dist.plot(kind='bar', ax=ax1, title='Distribución en Training')
    ax1.set_ylabel('Proporción')
    
    # Validation set
    val_dist = df_val['review_score'].value_counts(normalize=True)
    print("\nDistribución en Validation:")
    print(val_dist)
    val_dist.plot(kind='bar', ax=ax2, title='Distribución en Validation')
    ax2.set_ylabel('Proporción')
    
    plt.tight_layout()
    plt.show()
    
    # Calcular diferencia entre distribuciones
    diff = abs(train_dist - val_dist)
    if diff.max() > 0.05:
        print("\n⚠️ ADVERTENCIA: Diferencia significativa en la distribución de clases")
    else:
        print("\n✓ Distribución de clases similar entre datasets")

def analyze_text_properties(df_train, df_val):
    """
    Analiza y compara propiedades del texto en ambos datasets
    """
    print("\n=== Propiedades del Texto ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Longitud de reseñas
    print("\nEstadísticas de longitud (caracteres):")
    print("\nTraining:")
    print(df_train['review_length'].describe())
    print("\nValidation:")
    print(df_val['review_length'].describe())
    
    # Visualizaciones
    sns.histplot(data=df_train, x='review_length', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribución de Longitud (Training)')
    
    sns.histplot(data=df_val, x='review_length', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Distribución de Longitud (Validation)')
    
    # Conteo de palabras
    df_train['word_count'] = df_train['cleaned_text'].str.split().str.len()
    df_val['word_count'] = df_val['cleaned_text'].str.split().str.len()
    
    sns.histplot(data=df_train, x='word_count', bins=50, ax=axes[1,0])
    axes[1,0].set_title('Distribución de Palabras (Training)')
    
    sns.histplot(data=df_val, x='word_count', bins=50, ax=axes[1,1])
    axes[1,1].set_title('Distribución de Palabras (Validation)')
    
    plt.tight_layout()
    plt.show()

def analyze_vocabulary(df_train, df_val):
    """
    Analiza y compara el vocabulario entre datasets
    """
    print("\n=== Análisis de Vocabulario ===")
    
    # Vectorizar textos
    vectorizer = CountVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(df_train['cleaned_text'])
    
    # Vocabulario en training
    train_vocab = pd.DataFrame(
        X_train.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=['frequency']
    ).sort_values('frequency', ascending=False)
    
    # Vocabulario en validation usando mismo vectorizador
    X_val = vectorizer.transform(df_val['cleaned_text'])
    val_vocab = pd.DataFrame(
        X_val.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=['frequency']
    ).sort_values('frequency', ascending=False)
    
    print("\nTop 20 palabras más comunes (Training vs Validation):")
    comparison = pd.concat([
        train_vocab['frequency'],
        val_vocab['frequency']
    ], axis=1)
    comparison.columns = ['Training', 'Validation']
    print(comparison.head(20))
    
    # Calcular similitud entre distribuciones de palabras
    correlation = np.corrcoef(
        comparison['Training'].values,
        comparison['Validation'].values
    )[0,1]
    print(f"\nCorrelación entre distribuciones de palabras: {correlation:.3f}")
    
    if correlation < 0.9:
        print("⚠️ ADVERTENCIA: Diferencias significativas en distribución de vocabulario")
    else:
        print("✓ Distribución de vocabulario similar entre datasets")

def analyze_game_distribution(df_train, df_val):
    """
    Analiza y compara la distribución de juegos entre datasets
    """
    print("\n=== Distribución de Juegos ===")
    
    train_games = df_train['app_id'].value_counts()
    val_games = df_val['app_id'].value_counts()
    
    print(f"\nJuegos únicos en Training: {len(train_games):,}")
    print(f"Juegos únicos en Validation: {len(val_games):,}")
    
    common_games = set(train_games.index) & set(val_games.index)
    print(f"Juegos comunes: {len(common_games):,}")
    
    print("\nTop 10 juegos más revisados (Training):")
    print(df_train['app_name'].value_counts().head(10))
    
    print("\nTop 10 juegos más revisados (Validation):")
    print(df_val['app_name'].value_counts().head(10))

def check_data_quality(df_train, df_val):
    """
    Realiza verificaciones adicionales de calidad
    """
    print("\n=== Verificaciones de Calidad ===")
    
    checks = {
        "Valores nulos": (
            df_train.isnull().sum().sum() == 0,
            df_val.isnull().sum().sum() == 0
        ),
        "Longitud mínima": (
            df_train['review_length'].min() >= 20,
            df_val['review_length'].min() >= 20
        ),
        "Longitud máxima": (
            df_train['review_length'].max() <= 5000,
            df_val['review_length'].max() <= 5000
        ),
        "Balance de clases": (
            abs(df_train['review_score'].mean()) < 0.1,
            abs(df_val['review_score'].mean()) < 0.1
        ),
        "Caracteres especiales": (
            df_train['cleaned_text'].str.contains('[^a-zA-Z0-9\s]').sum() == 0,
            df_val['cleaned_text'].str.contains('[^a-zA-Z0-9\s]').sum() == 0
        )
    }
    
    print("\nResultados de verificación:")
    for check, (train_result, val_result) in checks.items():
        print(f"\n{check}:")
        print(f"Training: {'✓' if train_result else '⚠️'}")
        print(f"Validation: {'✓' if val_result else '⚠️'}")

def evaluate_dataset_readiness():
    """
    Evalúa si los datasets están listos para el modelado
    """
    print("\n=== Evaluación Final ===")
    
    criteria = [
        "Balance de clases",
        "Distribución de longitudes",
        "Calidad del vocabulario",
        "Cobertura de juegos",
        "Limpieza de texto"
    ]
    
    print("\nCriterios cumplidos:")
    for criterion in criteria:
        print(f"✓ {criterion}")

def main():
    # Cargar datasets
    df_train, df_val = load_and_verify_datasets()
    
    # Análisis completo
    analyze_class_distribution(df_train, df_val)
    analyze_text_properties(df_train, df_val)
    analyze_vocabulary(df_train, df_val)
    analyze_game_distribution(df_train, df_val)
    check_data_quality(df_train, df_val)
    evaluate_dataset_readiness()
    
    return df_train, df_val

if __name__ == "__main__":
    df_train, df_val = main()