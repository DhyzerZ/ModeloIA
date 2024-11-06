import pandas as pd
import numpy as np
import joblib

class GameReviewPredictor:
    def __init__(self, model_path='game_review_classifier.joblib'):
        # Cargar el modelo y el vectorizador guardados
        print("Cargando modelo desde:", model_path)
        model_components = joblib.load(model_path)
        self.vectorizer = model_components['vectorizer']
        self.model = model_components['model']
        print("Modelo cargado exitosamente!")
    
    def predict_sentiment(self, text):
        # Preprocesar y predecir una nueva reseña
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        return sentiment, probability

def main():
    try:
        # Inicializar el predictor
        predictor = GameReviewPredictor('game_review_classifier.joblib')
        
        # Interfaz de línea de comandos para procesar reseñas
        print("\n=== Predictor de Sentimientos para Reseñas de Juegos ===")
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
            sentiment, probs = predictor.predict_sentiment(user_review)
            
            # Mostrar resultados
            print("\nResultados del análisis:")
            print("-" * 25)
            print(f"Sentimiento predicho: {sentiment}")
            print(f"Nivel de confianza: {max(probs) * 100:.2f}%")
            print("\nProbabilidades por clase:")
            print(f"Negativo: {probs[0] * 100:.2f}%")
            print(f"Positivo: {probs[1] * 100:.2f}%")
            print("-" * 50)
            
    except FileNotFoundError:
        print("\nError: No se encontró el archivo del modelo.")
        print("Asegúrate de que el archivo 'game_review_classifier.joblib' está en el mismo directorio.")
    except Exception as e:
        print("\nError inesperado:", str(e))

if __name__ == "__main__":
    main()