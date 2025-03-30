import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import dataManager.DataExtractor as dataExtractor


def predict(input_texts: list[str]):
    """Predicts the topics for a list of input texts.

    Args:
        input_texts: List of strings to predict topics for
    """
    # Crear instancia del DataExtractor
    de = dataExtractor.DataExtractor()
    de.LEN_OF_TOKENS = np.load("data/x_data/test.npy").shape[1]

    # Cargar el modelo
    model = load_model("final_model.h5")

    for i, input_text in enumerate(input_texts):
        print(f'\n\n===== Prediction {i + 1}: "{input_text}" =====')

        # Procesar el texto de entrada
        filtered_input = de.filter_text(input_text)
        tokenized_input = de.tokenize_text(filtered_input)
        padded_input = de.pad_token(tokenized_input)

        # Convertir el dato a un array de numpy con batch size = 1
        padded_input = np.array([padded_input])

        # Hacer predicciones
        predictions = model.predict(padded_input, verbose=0)

        # Obtener los índices de los 5 temas con mayor probabilidad
        top5_indices = predictions[0].argsort()[-5:][::-1]
        top5_probs = predictions[0][top5_indices]

        # Mostrar los resultados
        print("\nTop 5 topics predicted:")
        print("-" * 40)
        print(f"{'Rank':<6}{'Topic':<30}{'Probability':<10}")
        print("-" * 40)
        for j, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
            tema = de.onehot_to_string(idx)  # Convertir índice a nombre del tema
            print(f"{j + 1:<6}{tema:<30}{prob:.4f}")


def main():
    input_texts = [
        "create a website using HTML, CSS, and JavaScript",
        "course on machine learning and data science with keras",
        "c, c++, python, java, javascript",
        "How to reach the happiness in life",
        "Mastering sql queries for data analysis and reporting",
        "3D modeling and animation with Blender",
    ]

    print("[INFO] Running predictions for multiple inputs")
    predict(input_texts)


if __name__ == "__main__":
    main()
