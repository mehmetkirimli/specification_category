#data_loader
#Veri setini indirip yüklemek için gerekli dosya

from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

def load_data():
    # CIFAR-10 veri setini yükle
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def show_samples(x_train):
    # İlk 10 resmi görselleştir
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i])
        plt.axis('off')
    plt.show()
