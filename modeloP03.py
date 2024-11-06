import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class GameReviewClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, 
                                        ngram_range=(1, 2),
                                        stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        
    def prepare_data(self, train_path, validation_path):
        # Cargar datos
        self.train_data = pd.read_csv(train_path)
        self.validation_data = pd.read_csv(validation_path)
        
        # Preparar textos y etiquetas
        X_train = self.vectorizer.fit_transform(self.train_data['cleaned_text'])
        y_train = self.train_data['review_score']
        
        X_val = self.vectorizer.transform(self.validation_data['cleaned_text'])
        y_val = self.validation_data['review_score']
        
        return X_train, X_val, y_train, y_val
    
    def train(self, X_train, y_train):
        # Entrenar el modelo
        print("Entrenando el modelo...")
        self.model.fit(X_train, y_train)
        print("¡Entrenamiento completado!")
    
    def evaluate(self, X_val, y_val):
        # Evaluar el modelo
        print("\nEvaluando el modelo...")
        predictions = self.model.predict(X_val)
        
        # Imprimir métricas
        print("\nInforme de Clasificación:")
        print(classification_report(y_val, predictions))
        
        # Matriz de confusión
        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_val, predictions))
        
        return predictions
    
    def save_model(self, model_path='game_review_classifier.joblib'):
        # Guardar el modelo y el vectorizador
        print(f"\nGuardando modelo en {model_path}...")
        model_components = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        joblib.dump(model_components, model_path)
        print("¡Modelo guardado exitosamente!")

def main():
    # Inicializar el clasificador
    classifier = GameReviewClassifier()
    
    try:
        # Preparar datos
        print("Preparando datos...")
        X_train, X_val, y_train, y_val = classifier.prepare_data('train_dataset.csv', 
                                                                'validation_dataset.csv')
        
        # Entrenar modelo
        classifier.train(X_train, y_train)
        
        # Evaluar modelo
        classifier.evaluate(X_val, y_val)
        
        # Guardar modelo
        classifier.save_model()
        
        print("\n¡Proceso completado!")
        print("Puedes usar el modelo guardado 'game_review_classifier.joblib' para hacer predicciones.")
        
    except FileNotFoundError as e:
        print("\nError: No se encontraron los archivos de datos.")
        print("Asegúrate de tener 'train_dataset.csv' y 'validation_dataset.csv' en el directorio.")
    except Exception as e:
        print("\nError inesperado:", str(e))

if __name__ == "__main__":
    main()