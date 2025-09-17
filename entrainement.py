import tensorflow as tf
from tensorflow.keras import layers, models
import os
import shutil
from pathlib import Path

# CONFIG
original_data_paths = {
    'live': [
        'data/live/faces_from_photos',
        'data/live/faces_from_videos'
    ],
    'spoof': [
        'data/spoof/faces_from_photo',
        'data/spoof/faces_from_videos'
    ]
}
train_dir = 'train_dataset'
img_size = (224, 224)
batch_size = 32
epochs = 10

# ----- ÉTAPE 1 : Nettoyer et reconstruire train_dataset -----
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(os.path.join(train_dir, 'live'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'spoof'), exist_ok=True)

# ----- ÉTAPE 2 : Copier les images -----
for label in original_data_paths:
    for source_folder in original_data_paths[label]:
        if not os.path.exists(source_folder):
            continue
        for file in os.listdir(source_folder):
            src_path = os.path.join(source_folder, file)
            # Vérifie que c'est bien une image
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                dest_path = os.path.join(train_dir, label, file)
                shutil.copy(src_path, dest_path)

print(f"[INFO] Données copiées vers {train_dir}/live et spoof")

# ----- ÉTAPE 3 : Préparation des données -----
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

print("[INFO] Classes :", train_generator.class_indices)

# ----- ÉTAPE 4 : Création du modèle -----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # sortie binaire : 1 = spoof, 0 = live
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ----- ÉTAPE 5 : Entraînement -----
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# ÉTAPE 6 : Sauvegarde du modèle
model.save('anti_spoof_model.h5')
print("[INFO] Modèle sauvegardé sous 'anti_spoof_model.h5'")
