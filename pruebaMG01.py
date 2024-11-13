import pandas as pd
import numpy as np
import joblib
import re

class GameReviewPredictor:
    def __init__(self, model_path='game_review_classifier.joblib'):
        # Cargar el modelo y el vectorizador guardados
        print("Cargando modelo desde:", model_path)
        model_components = joblib.load(model_path)
        self.vectorizer = model_components['vectorizer']
        self.model = model_components['model']
        print("Modelo cargado exitosamente!")
    
    def predict_sentiment_and_problems(self, text):
        # Preprocesar y predecir una nueva reseña
        text_vectorized = self.vectorizer.transform([text])
        sentiment_prediction = self.model.predict(text_vectorized)[0]
        sentiment_probability = self.model.predict_proba(text_vectorized)[0]
        
        problem_categories, problem_probs = self.classify_problems(text)

        sentiment = "Positivo" if sentiment_prediction == 1 else "Negativo"
        return sentiment, sentiment_probability, problem_categories, problem_probs

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

def main():
    try:
        # Inicializar el predictor
        predictor = GameReviewPredictor('game_review_classifier.joblib')
        
        # Interfaz de línea de comandos para procesar reseñas
        print("\n=== Predictor de Sentimientos y Problemas para Reseñas de Juegos ===")
        print("Escribe 'exit' para salir")
        print("-" * 50)
        
        while True:
            # Solicitar reseña al usuario
            user_review = input("\nIngresa una reseña del juego: ")
            
            # Verificar si el usuario quiere salir
            if user_review.lower() == 'exit':
                print("\n¡Gracias por usar el modelo!")
                break
            
            # Realizar predicción
            sentiment, sentiment_probability, problem_categories, problem_probs = predictor.predict_sentiment_and_problems(user_review)
            
            # Mostrar resultados
            print("\nResultados del análisis:")
            print("-" * 25)
            print(f"Sentimiento predicho: {sentiment}")
            print(f"Nivel de confianza: {max(sentiment_probability) * 100:.2f}%")
            print("\nProbabilidades por clase de sentimiento:")
            print(f"Negativo: {sentiment_probability[0] * 100:.2f}%")
            print(f"Positivo: {sentiment_probability[1] * 100:.2f}%")
            
            if problem_categories:
                print("\nProblemas detectados y probabilidades:")
                for i, problem in enumerate(problem_categories):
                    print(f"- {problem}: {problem_probs[i] * 100:.2f}%")
            else:
                print("\nNo se detectaron problemas específicos en la reseña.")

            print("-" * 50)
            
    except FileNotFoundError:
        print("\nError: No se encontró el archivo del modelo.")
        print("Asegúrate de que el archivo 'game_review_classifier.joblib' está en el mismo directorio.")
    except Exception as e:
        print("\nError inesperado:", str(e))

if __name__ == "__main__":
    main()