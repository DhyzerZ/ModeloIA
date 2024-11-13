import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

class GameReviewClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, 
                                        ngram_range=(1, 2),
                                        stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        
    def classify_problems(self, text):
        """
        Clasifica los problemas mencionados en el texto de la reseña.
        Devuelve una lista de categorías de problemas y sus probabilidades.
        """
        problem_categories = []
        problem_probs = []

        # Problemas técnicos
        if re.search(r'(performance|bug|optimization|crash|lag)', text, re.IGNORECASE):
            problem_categories.append('Problemas técnicos')
            problem_probs.append(0.8)
        else:
            problem_probs.append(0.2)

        # Problemas de diseño
        if re.search(r'(gameplay|ui|ux|controls)', text, re.IGNORECASE):
            problem_categories.append('Problemas de diseño')
            problem_probs.append(0.7)
        else:
            problem_probs.append(0.3)

        # Problemas de monetización
        if re.search(r'(price|microtransaction|dlc|pay-to-win)', text, re.IGNORECASE):
            problem_categories.append('Problemas de monetización')
            problem_probs.append(0.6)
        else:
            problem_probs.append(0.4)

        # Problemas de contenido
        if re.search(r'(story|content|variety)', text, re.IGNORECASE):
            problem_categories.append('Problemas de contenido')
            problem_probs.append(0.7)
        else:
            problem_probs.append(0.3)

        return problem_categories, problem_probs
        
    def prepare_data(self, train_path, validation_path):
        # Cargar datos
        self.train_data = pd.read_csv(train_path)
        self.validation_data = pd.read_csv(validation_path)
        
        # Preparar textos, etiquetas y categorías de problemas
        X_train = self.vectorizer.fit_transform(self.train_data['cleaned_text'])
        y_train = self.train_data['review_score']
        self.train_data['problem_categories'], self.train_data['problem_probs'] = zip(*self.train_data['cleaned_text'].apply(self.classify_problems))

        X_val = self.vectorizer.transform(self.validation_data['cleaned_text'])
        y_val = self.validation_data['review_score']
        self.validation_data['problem_categories'], self.validation_data['problem_probs'] = zip(*self.validation_data['cleaned_text'].apply(self.classify_problems))

        return X_train, X_val, y_train, y_val

    def train(self, X_train, y_train):
        # Entrenar el modelo
        print("Entrenando el modelo...")
        self.model.fit(X_train, y_train)
        print("¡Entrenamiento completado!")

    def evaluate(self, X_val, y_val):
        # Evaluar el modelo
        print("\nEvaluando el modelo...")
        sentiment_predictions = self.model.predict(X_val)
        
        # Imprimir métricas de sentimiento
        print("\nInforme de Clasificación de Sentimiento:")
        print(classification_report(y_val, sentiment_predictions))

        # Matriz de confusión de sentimiento
        print("\nMatriz de Confusión de Sentimiento:")
        print(confusion_matrix(y_val, sentiment_predictions))

        # Analizar problemas específicos
        problem_predictions = self.validation_data.apply(lambda row: self.classify_problems(row['cleaned_text']), axis=1)
        self.validation_data['predicted_problem_categories'], self.validation_data['predicted_problem_probs'] = zip(*problem_predictions)
        self.analyze_problems(self.validation_data)

        return sentiment_predictions

    def analyze_problems(self, data):
        print("\nAnálisis de problemas específicos:")
        problem_counts = data['predicted_problem_categories'].explode().value_counts()
        for problem, count in problem_counts.items():
            print(f"{problem}: {count} reseñas")

        print("\nProbabilidades promedio de los problemas:")
        # Convertir la lista de probabilidades en un DataFrame para facilitar el cálculo
        probs_df = pd.DataFrame(data['predicted_problem_probs'].tolist())
        problem_types = ['Problemas técnicos', 'Problemas de diseño', 
                        'Problemas de monetización', 'Problemas de contenido']
        
        for i, problem in enumerate(problem_types):
            avg_prob = probs_df[i].mean() * 100
            print(f"{problem}: {avg_prob:.2f}%")

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