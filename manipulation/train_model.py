# Modeli eğitmek için
# Bu dosyada modeli eğitmek için bir fnk yazacağız.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from cifar_data import load_data
from preprocess_data import normalize_data
from cnn_model import create_model

def train_model():
    # Veri setini yükle ve normalleştir
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = normalize_data(x_train, x_test)

    # Etiketleri kategorik hale getir
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Modeli oluştur ve derle
    model = create_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğit
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    
    # Eğitilmiş modeli kaydet
    model.save("cifar10_model.h5")

if __name__ == "__main__":
    train_model()
