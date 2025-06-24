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
    shuffle=False  
)

val_generator = val_datagen.flow_from_directory(
    base_path +'/valid',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  
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

# --- Custom CNN Classification Report and Confusion Matrix ---
y_true = test_generator.classes
y_pred_cnn = np.argmax(cnn.predict(test_generator), axis=1)
print("\nCustom CNN Classification Report:")
print(classification_report(y_true, y_pred_cnn, target_names=class_names))
print("Custom CNN Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_cnn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Custom CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

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

# --- VGG16 Classification Report and Confusion Matrix ---
y_pred_vgg16 = np.argmax(vgg16.predict(test_generator), axis=1)
print("\nVGG16 Classification Report:")
print(classification_report(y_true, y_pred_vgg16, target_names=class_names))
print("VGG16 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_vgg16)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('VGG16 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n--- Training ResNet50 Model ---\n")
resnet50, history_resnet50_stage1, history_resnet50_stage2 = create_model(ResNet50, (img_height, img_width, 3), len(class_names))

print("\n--- ResNet50 Model Summary ---\n")
resnet50.summary()

print("\n--- Evaluating ResNet50 Model ---\n")
resnet50_loss, resnet50_acc = resnet50.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', resnet50_acc)

# --- ResNet50 Classification Report and Confusion Matrix ---
y_pred_resnet50 = np.argmax(resnet50.predict(test_generator), axis=1)
print("\nResNet50 Classification Report:")
print(classification_report(y_true, y_pred_resnet50, target_names=class_names))
print("ResNet50 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_resnet50)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n--- Training InceptionV3 Model ---\n")
inceptionv3, history_inceptionv3_stage1, history_inceptionv3_stage2 = create_model(InceptionV3, (img_height, img_width, 3), len(class_names))

print("\n--- InceptionV3 Model Summary ---\n")
inceptionv3.summary()

print("\n--- Evaluating InceptionV3 Model ---\n")
inceptionv3_loss, inceptionv3_acc = inceptionv3.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', inceptionv3_acc)

# --- InceptionV3 Classification Report and Confusion Matrix ---
y_pred_inceptionv3 = np.argmax(inceptionv3.predict(test_generator), axis=1)
print("\nInceptionV3 Classification Report:")
print(classification_report(y_true, y_pred_inceptionv3, target_names=class_names))
print("InceptionV3 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_inceptionv3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('InceptionV3 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n--- Training MobileNetV2 Model ---\n")
mobilenetv2,history_mobilenetv2_stage1,history_mobilenetv2_stage2 = create_model(MobileNetV2, (img_height, img_width, 3), len(class_names))

print("\n--- MobileNetV2 Model Summary ---\n")
mobilenetv2.summary()

print("\n--- Evaluating MobileNetV2 Model ---\n")
mobilenetv2_loss, mobilenetv2_acc = mobilenetv2.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', mobilenetv2_acc)

# --- MobileNetV2 Classification Report and Confusion Matrix ---
y_pred_mobilenetv2 = np.argmax(mobilenetv2.predict(test_generator), axis=1)
print("\nMobileNetV2 Classification Report:")
print(classification_report(y_true, y_pred_mobilenetv2, target_names=class_names))
print("MobileNetV2 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_mobilenetv2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('MobileNetV2 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n--- Training EfficientNetB0 Model ---\n")
efficientnetb0,history_efficientnetb0_stage1,history_efficientnetb0_stage2 = create_model(EfficientNetB0, (img_height, img_width, 3), len(class_names))

print("\n--- EfficientNetB0 Model Summary ---\n")
efficientnetb0.summary()

print("\n--- Evaluating EfficientNetB0 Model ---\n")
efficientnetb0_loss, efficientnetb0_acc = efficientnetb0.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', efficientnetb0_acc)

# --- EfficientNetB0 Classification Report and Confusion Matrix ---
y_pred_efficientnetb0 = np.argmax(efficientnetb0.predict(test_generator), axis=1)
print("\nEfficientNetB0 Classification Report:")
print(classification_report(y_true, y_pred_efficientnetb0, target_names=class_names))
print("EfficientNetB0 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_efficientnetb0)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('EfficientNetB0 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n=== Model Performance Comparison ===")
print("{:<20} {:<15} {:<15}".format("Model", "Test Accuracy", "Test Loss"))
print("-" * 50)
print("{:<20} {:<15.4f} {:<15.4f}".format("Custom CNN", cnn_acc, cnn_loss))
print("{:<20} {:<15.4f} {:<15.4f}".format("VGG16", vgg16_acc, vgg16_loss))
print("{:<20} {:<15.4f} {:<15.4f}".format("ResNet50", resnet50_acc, resnet50_loss))
print("{:<20} {:<15.4f} {:<15.4f}".format("InceptionV3", inceptionv3_acc, inceptionv3_loss))
print("{:<20} {:<15.4f} {:<15.4f}".format("MobileNetV2", mobilenetv2_acc, mobilenetv2_loss))
print("{:<20} {:<15.4f} {:<15.4f}".format("EfficientNetB0", efficientnetb0_acc, efficientnetb0_loss))

models = ["Custom CNN", "VGG16", "ResNet50", "InceptionV3", "MobileNetV2", "EfficientNetB0"]
accuracies = [cnn_acc, vgg16_acc, resnet50_acc, inceptionv3_acc, mobilenetv2_acc, efficientnetb0_acc]

plt.figure(figsize=(10,5))
plt.bar(models, accuracies, color='skyblue')
plt.ylabel('Test Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()


# Custom CNN
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Custom CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Custom CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# VGG16
acc = history_vgg16_stage1.history['accuracy'] + history_vgg16_stage2.history['accuracy']
val_acc = history_vgg16_stage1.history['val_accuracy'] + history_vgg16_stage2.history['val_accuracy']
loss = history_vgg16_stage1.history['loss'] + history_vgg16_stage2.history['loss']
val_loss = history_vgg16_stage1.history['val_loss'] + history_vgg16_stage2.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('VGG16 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('VGG16 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ResNet50
acc = history_resnet50_stage1.history['accuracy'] + history_resnet50_stage2.history['accuracy']
val_acc = history_resnet50_stage1.history['val_accuracy'] + history_resnet50_stage2.history['val_accuracy']
loss = history_resnet50_stage1.history['loss'] + history_resnet50_stage2.history['loss']
val_loss = history_resnet50_stage1.history['val_loss'] + history_resnet50_stage2.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('ResNet50 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('ResNet50 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# InceptionV3
acc = history_inceptionv3_stage1.history['accuracy'] + history_inceptionv3_stage2.history['accuracy']
val_acc = history_inceptionv3_stage1.history['val_accuracy'] + history_inceptionv3_stage2.history['val_accuracy']
loss = history_inceptionv3_stage1.history['loss'] + history_inceptionv3_stage2.history['loss']
val_loss = history_inceptionv3_stage1.history['val_loss'] + history_inceptionv3_stage2.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('InceptionV3 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('InceptionV3 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# MobileNetV2
acc = history_mobilenetv2_stage1.history['accuracy'] + history_mobilenetv2_stage2.history['accuracy']
val_acc = history_mobilenetv2_stage1.history['val_accuracy'] + history_mobilenetv2_stage2.history['val_accuracy']
loss = history_mobilenetv2_stage1.history['loss'] + history_mobilenetv2_stage2.history['loss']
val_loss = history_mobilenetv2_stage1.history['val_loss'] + history_mobilenetv2_stage2.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('MobileNetV2 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# EfficientNetB0
acc = history_efficientnetb0_stage1.history['accuracy'] + history_efficientnetb0_stage2.history['accuracy']
val_acc = history_efficientnetb0_stage1.history['val_accuracy'] + history_efficientnetb0_stage2.history['val_accuracy']
loss = history_efficientnetb0_stage1.history['loss'] + history_efficientnetb0_stage2.history['loss']
val_loss = history_efficientnetb0_stage1.history['val_loss'] + history_efficientnetb0_stage2.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('EfficientNetB0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('EfficientNetB0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()



