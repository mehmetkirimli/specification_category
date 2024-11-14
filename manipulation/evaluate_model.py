# Modeli değerlendirmek için
# Bu dosyada eğitim sonucunda kaydedilen modeli yükleyip test seti üzerinde değerlendireceğiz.

from tensorflow.keras.models import load_model
from cifar_data import load_data
from preprocess_data import normalize_data
from tensorflow.keras.utils import to_categorical

def evaluate_model():
    # Veri setini yükle ve normalleştir
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = normalize_data(x_train, x_test)

    # Etiketleri kategorik hale getir
    y_test = to_categorical(y_test, 10)

    # Eğitilmiş modeli yükle
    model = load_model("cifar10_model.h5")

    # Modeli değerlendir
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_model()
