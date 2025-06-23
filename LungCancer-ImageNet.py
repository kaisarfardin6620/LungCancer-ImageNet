import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Input
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau


base_path = 'Data'

img_height, img_width = 224, 224
batch_size = 16
epochs = 30

early_stopping = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1]
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    base_path +'/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    base_path +'/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Changed to False for reproducibility
)

val_generator = val_datagen.flow_from_directory(
    base_path +'/valid',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Changed to False for reproducibility
)

print('Number of training samples:', train_generator.samples)
print('Number of testing samples:', test_generator.samples)
print('Number of validation samples:', val_generator.samples)

print('Class indices for traning data:', train_generator.class_indices)

images, labels = next(train_generator)

plt.figure(figsize=(10, 10))
class_names = list(train_generator.class_indices.keys())
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(images[i])
  plt.title(class_names[labels[i].argmax()])
  plt.axis('off')
plt.tight_layout()
plt.show()

def plot_class_distribution(generator, title):
    class_counts = generator.classes
    class_labels = list(generator.class_indices.keys())
    unique_classes, counts = np.unique(class_counts, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(class_labels, counts)
    plt.title(f'Class Distribution - {title}')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_class_distribution(train_generator, 'Training Set')
plot_class_distribution(test_generator, 'Test Set')
plot_class_distribution(val_generator, 'Validation Set')

cnn = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_labels = train_generator.classes
class_indices = np.unique(train_labels)
class_weight_values = compute_class_weight(class_weight='balanced', classes=class_indices, y=train_labels)
class_weight = dict(zip(class_indices, class_weight_values))
print('Computed class weights:', class_weight)

print("\n--- Training Custom CNN Model ---\n")
history = cnn.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping,reduce_lr],
    class_weight=class_weight
)

print("\n--- Evaluating Custom CNN Model ---\n")
cnn_loss, cnn_acc = cnn.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', cnn_acc)


def create_model(base_model_class, input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    base_model = base_model_class(weights='imagenet', include_top=False)
    base_model.trainable = False
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Stage 1: Train top layers
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history_stage1 = model.fit(train_generator, epochs=epochs, validation_data=val_generator, 
                               callbacks=[early_stopping, reduce_lr], class_weight=class_weight)
    
    # Stage 2: Unfreeze and fine-tune
    fine_tune_from = len(base_model.layers) - 20
    for layer in base_model.layers[fine_tune_from:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history_stage2 = model.fit(train_generator, epochs=epochs, validation_data=val_generator, 
                               callbacks=[early_stopping, reduce_lr], class_weight=class_weight)
    
    return model, history_stage1, history_stage2

print("\n--- Training VGG16 Model ---\n")
vgg16, history_vgg16_stage1, history_vgg16_stage2 = create_model(VGG16, (img_height, img_width, 3), len(class_names))

print("\n--- VGG16 Model Summary ---\n")
vgg16.summary()

print("\n--- Evaluating VGG16 Model ---\n")
vgg16_loss, vgg16_acc = vgg16.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', vgg16_acc)

print("\n--- Training ResNet50 Model ---\n")
resnet50, history_resnet50_stage1, history_resnet50_stage2 = create_model(ResNet50, (img_height, img_width, 3), len(class_names))

print("\n--- ResNet50 Model Summary ---\n")
resnet50.summary()

print("\n--- Evaluating ResNet50 Model ---\n")
resnet50_loss, resnet50_acc = resnet50.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', resnet50_acc)

print("\n--- Training InceptionV3 Model ---\n")
inceptionv3, history_inceptionv3_stage1, history_inceptionv3_stage2 = create_model(InceptionV3, (img_height, img_width, 3), len(class_names))

print("\n--- InceptionV3 Model Summary ---\n")
inceptionv3.summary()

print("\n--- Evaluating InceptionV3 Model ---\n")
inceptionv3_loss, inceptionv3_acc = inceptionv3.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', inceptionv3_acc)

print("\n--- Training MobileNetV2 Model ---\n")
mobilenetv2,history_mobilenetv2_stage1,history_mobilenetv2_stage2 = create_model(MobileNetV2, (img_height, img_width, 3), len(class_names))

print("\n--- MobileNetV2 Model Summary ---\n")
mobilenetv2.summary()

print("\n--- Evaluating MobileNetV2 Model ---\n")
mobilenetv2_loss, mobilenetv2_acc = mobilenetv2.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', mobilenetv2_acc)

print("\n--- Training EfficientNetB0 Model ---\n")
efficientnetb0,history_efficientnetb0_stage1,history_efficientnetb0_stage2 = create_model(EfficientNetB0, (img_height, img_width, 3), len(class_names))

print("\n--- EfficientNetB0 Model Summary ---\n")
efficientnetb0.summary()

print("\n--- Evaluating EfficientNetB0 Model ---\n")
efficientnetb0_loss, efficientnetb0_acc = efficientnetb0.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', efficientnetb0_acc)



