import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator




train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    directory=r'C:\Users\Nithishraj\Desktop\dataset', 
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

validation_dataset = validation_datagen.flow_from_directory(
    directory=r'C:\Users\Nithishraj\Desktop\validation',  
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=train_dataset.samples // train_dataset.batch_size,
                      epochs=30,
                      validation_data=validation_dataset)


print("Class indices:", validation_dataset.class_indices)


model_save_path = 'D:/path_to_your_model/your_model.keras'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True) 
model.save(model_save_path)
print(f"Model saved to {model_save_path}") 
