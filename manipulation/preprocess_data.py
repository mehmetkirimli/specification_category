# Veriyi işlemek için (örneğin normalleştirme)
# Bu dosyada veri setindeki piksel değerlerini 0-255 aralığından 0-1 aralığına getireceğiz.

def normalize_data(x_train, x_test):
    # Piksel değerlerini 0-1 aralığına getir
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test
