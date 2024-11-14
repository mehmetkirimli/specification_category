# Bu dosya modeli kullanılarak birkaç test görüntüsünü sınıflandırıp tahminleri gerçek etiketlerle birlikte görselleştireceğiz.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from cifar_data import load_data
from preprocess_data import normalize_data
from tensorflow.keras.utils import to_categorical

# CIFAR-10 sınıf etiketleri
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_predictions():
    # Veri setini yükle ve normalleştir
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = normalize_data(x_train, x_test)
    
    # Modeli yükle
    model = load_model("cifar10_model.h5")
    
    # Test verilerinden birkaç örnek seç ve tahmin yap
    indices = np.random.choice(len(x_test), size=10, replace=False)
    x_sample = x_test[indices]
    y_true = y_test[indices]
    
    # Modelin tahminleri
    predictions = model.predict(x_sample)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_true.flatten()

    # Görselleri tahminlerle birlikte göster
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_sample[i])
        plt.title(f"Gerçek: {class_names[y_true[i]]}\nTahmin: {class_names[y_pred[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_predictions()
