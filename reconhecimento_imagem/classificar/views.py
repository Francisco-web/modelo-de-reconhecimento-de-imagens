from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Caminho do modelo treinado
MODEL_PATH = 'classifier_model.h5'

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ['Cão', 'Pessoa', 'Carro']  # Categorias

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Guardar a imagem enviada
        image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(f'uploads/{image.name}', image)

        # Carregar a imagem para previsão
        image_full_path = os.path.join(fs.location, image_path)
        img = load_img(image_full_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        # Prever a categoria
        predictions = model.predict(img_array)
        predicted_class = CLASSES[predictions.argmax()]

        return render(request, 'classificar/index.html', {
            'image_url': fs.url(image_path),
            'predicted_class': predicted_class
        })

    return render(request, 'classificar/index.html')
