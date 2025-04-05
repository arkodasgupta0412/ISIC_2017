# load required modules and dataset

import kagglehub, os, tensorflow, keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Download latest version
dataset_path = kagglehub.dataset_download("sujansarkar/isic-2017-preprocessed-augmented")
# print("Path to dataset files:", dataset_path
data_dir = os.path.join(dataset_path, "content/Linear_Exact_Aug")


# load the dataset => training, cross-validation and test sets

train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
valid_dir = os.path.join(data_dir, "Valid")

# print("Train Directory:", train_dir)
# print("Test Directory:", test_dir)
# print("Validation Directory:", valid_dir)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


base_model = ConvNeXtSmall(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Fully Connected Layers with SELU activation
x = Dense(512, activation='selu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='selu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32, activation='selu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

# Output Layer
x = Dense(3, activation='softmax', kernel_regularizer=l2(0.01))(x)

model = Model(inputs=base_model.input, outputs=x)

#model.summary()

# compiling the model
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler: Reduce LR if validation loss stops improving
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Early Stopping: Stop training if val_loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)


# fitting the model
EPOCHS = 20

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)


# test loss and test accurracy
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")


import matplotlib.pyplot as plt
import numpy as np

epochs = range(1, len(history.history['accuracy']) + 1)

epoch_ticks = np.arange(5, len(epochs) + 1, 5)

y_tick_interval_acc = 0.1
y_tick_interval_loss = 2

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, history.history['accuracy'], 'bo-', label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'ro-', label='Val Accuracy')
plt.axhline(y=test_accuracy, color='g', linestyle='--', label='Test Accuracy')
plt.xticks(epoch_ticks)
plt.yticks(np.arange(0, 1.1, y_tick_interval_acc))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation vs Test Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 3, 2)
plt.plot(epochs, history.history['loss'], 'bo-', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'ro-', label='Val Loss')
plt.axhline(y=test_loss, color='g', linestyle='--', label='Test Loss')
plt.xticks(epoch_ticks) 
plt.yticks(np.arange(0, max(history.history['loss'] + history.history['val_loss']) + y_tick_interval_loss, y_tick_interval_loss))  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation vs Test Loss')
plt.legend()
plt.grid(True)