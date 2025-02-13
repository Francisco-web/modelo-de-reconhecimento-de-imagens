import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho do conjunto de dados
DATASET_PATH = "dataset/"  # Atualize para o caminho do seu conjunto de dados
MODEL_PATH = "classifier_model.h5"

# Configuração do gerador de dados
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Dados de treino e validação
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Criação do modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # Três categorias: Cão, Pessoa, Carro
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(train_data, validation_data=val_data, epochs=10)

# Salvar o modelo treinado
model.save(MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")
