import numpy as np

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

import matplotlib.pyplot as plt


def load_data():
    NP_X_TRAIN = "data/x_data/train.npy"
    NP_Y_TRAIN = "data/y_data/train.npy"

    NP_X_VAL = "data/x_data/val.npy"
    NP_Y_VAL = "data/y_data/val.npy"

    NP_X_TEST = "data/x_data/test.npy"
    NP_Y_TEST = "data/y_data/test.npy"

    x_train = np.load(NP_X_TRAIN)
    y_train = np.load(NP_Y_TRAIN)

    x_val = np.load(NP_X_VAL)
    y_val = np.load(NP_Y_VAL)

    x_test = np.load(NP_X_TEST)
    y_test = np.load(NP_Y_TEST)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_model(input_shape, output_shape):
    model = Sequential(
        [
            Dense(512, activation="relu", input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(output_shape, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=18):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    return model, history


def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # For multi-label classification, additional metrics might be useful
    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.8).astype(int)

    # Calculate the exact match ratio (all labels correct)
    exact_match = np.mean(np.all(y_pred_binary == y_test, axis=1))
    print(f"Exact Match Ratio: {exact_match:.4f}")

    # Calculate top-5 accuracy
    evaluate_top_n_accuracy(model, x_test, y_test)

    return y_pred_binary


def evaluate_top_n_accuracy(model, x_test, y_test, n=5):
    y_pred = model.predict(x_test)

    top5_pred_indices = np.argsort(y_pred, axis=1)[:, -5:]

    y_true_indices = [np.where(y_test[i] == 1)[0] for i in range(y_test.shape[0])]

    correct_predictions = sum(
        any(idx in top5_pred_indices[i] for idx in y_true_indices[i])
        for i in range(len(y_test))
    )

    top5_accuracy = correct_predictions / len(y_test)

    print(f"\nTop-{n} Accuracy: {top5_accuracy:.4f}\n")

    return top5_accuracy


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]
    model = create_model(input_shape, output_shape)

    model.summary()

    model, history = train_model(model, x_train, y_train, x_val, y_val)

    plot_training_history(history)

    y_pred = evaluate_model(model, x_test, y_test)

    model.save("final_model.h5")
    print("Model saved to 'final_model.h5'")


if __name__ == "__main__":
    main()
