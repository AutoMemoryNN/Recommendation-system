import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import dataManager.DataExtractor as dataExtractor


def predict(input_text: str):
    """Predicts the topics of the input text."""
    # Crear instancia del DataExtractor y procesar el texto de entrada
    de = dataExtractor.DataExtractor()
    filtered_input = de.filter_text(input_text)
    tokenized_input = de.tokenize_text(filtered_input)
    de.LEN_OF_TOKENS = 60
    padded_input = de.pad_token(tokenized_input)

    # Convertir el dato a un array de numpy con batch size = 1
    padded_input = np.array([padded_input])

    # Cargar el modelo
    model = load_model("final_model.h5")

    # Hacer predicciones
    predictions = model.predict(padded_input)

    # Obtener los índices de los 5 temas con mayor probabilidad
    top5_indices = predictions[0].argsort()[-5:][::-1]
    top5_probs = predictions[0][top5_indices]

    # Mostrar los resultados
    print("\nTop 5 temas predichos:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        tema = de.onehot_to_string(idx)  # Convertir índice a nombre del tema
        print(f"{i + 1}. {tema}: {prob:.4f}")


def main():
    input_text = "I am a software developer and I like to play soccer."
    print("[INFO] Running prediction for input:", input_text)
    predict(input_text)


if __name__ == "__main__":
    main()
