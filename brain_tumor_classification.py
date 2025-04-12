import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, Xception, NASNetMobile, NASNetLarge
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Define dataset paths
data_dir = "archive"
train_dir = os.path.join(data_dir, "training")
test_dir = os.path.join(data_dir, "testing")
# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)
# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
# Define base models
def build_model(base_model):
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)  # 4 classes
    model = Model(inputs, outputs)
    return model
# Instantiate models
models = {
    "VGG16": build_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    "InceptionV3": build_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    "ResNet50": build_model(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    "Xception": build_model(Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    "NASNetMobile": build_model(NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    "NASNetLarge": build_model(NASNetLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
}
# Compile models
for name, model in models.items():
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"{name} model compiled successfully.")
# Train models
history = {}
for name, model in models.items():
    print(f"Training {name} model...")
    checkpoint = ModelCheckpoint(f"best_{name}.keras", save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history[name] = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )

print("Training complete.")
# Evaluate models on test data
evaluation_results = {}
for name in models.keys():
    print(f"Evaluating {name} model...")
    model = load_model(f"best_{name}.keras")
    loss, accuracy = model.evaluate(test_generator)
    evaluation_results[name] = {'loss': loss, 'accuracy': accuracy}
    print(f"{name} - Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

print("Evaluation complete.")
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Evaluate models on test data with additional metrics
evaluation_results = {}
y_true = test_generator.classes  # True labels
class_labels = list(test_generator.class_indices.keys())

for name in models.keys():
    print(f"Evaluating {name} model...")
    model = load_model(f"best_{name}.keras")
    y_pred_probs = model.predict(test_generator)  # Predicted probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels

    # Compute metrics
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_probs, multi_class="ovr")
    loss = log_loss(y_true, y_pred_probs)  # Compute loss

    evaluation_results[name] = {
        'accuracy': report['accuracy'],
        'precision': {cls: report[cls]['precision'] for cls in class_labels},
        'recall': {cls: report[cls]['recall'] for cls in class_labels},
        'f1-score': {cls: report[cls]['f1-score'] for cls in class_labels},
        'auc': auc_score,
        'loss': loss
    }

    print(f"{name} - Accuracy: {report['accuracy']:.4f}, AUC: {auc_score:.4f}, Loss: {loss:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

print("Evaluation complete.")

# Extracting metrics for plotting
model_names = list(evaluation_results.keys())

accuracy_scores = [evaluation_results[name]['accuracy'] for name in model_names]
auc_scores = [evaluation_results[name]['auc'] for name in model_names]
loss_scores = [evaluation_results[name]['loss'] for name in model_names]

# Extracting class-wise precision, recall, f1-score (averaged over all classes)
precision_scores = [np.mean(list(evaluation_results[name]['precision'].values())) for name in model_names]
recall_scores = [np.mean(list(evaluation_results[name]['recall'].values())) for name in model_names]
f1_scores = [np.mean(list(evaluation_results[name]['f1-score'].values())) for name in model_names]

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, accuracy_scores, marker='o', linestyle='-', color='blue', label="Accuracy")
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# AUC Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, auc_scores, marker='s', linestyle='-', color='purple', label="AUC Score")
plt.xlabel('Models')
plt.ylabel('AUC Score')
plt.title('Model AUC Score Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, loss_scores, marker='d', linestyle='-', color='brown', label="Loss")
plt.xlabel('Models')
plt.ylabel('Loss')
plt.title('Model Loss Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# Precision Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, precision_scores, marker='o', linestyle='-', color='green', label="Precision")
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Model Precision Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# Recall Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, recall_scores, marker='s', linestyle='-', color='orange', label="Recall")
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Model Recall Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# F1-Score Graph
plt.figure(figsize=(8, 5))
plt.plot(model_names, f1_scores, marker='^', linestyle='-', color='red', label="F1-Score")
plt.xlabel('Models')
plt.ylabel('F1-Score')
plt.title('Model F1-Score Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
# Load Best Model for Real-World Inference
best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
print(f"Best model: {best_model_name}")
best_model = load_model(f"best_{best_model_name}.keras")
def predict_image(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = best_model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

#Example usage:
result = predict_image("archive\Testing\meningioma\Te-me_0010.jpg")
print(f"Predicted Tumor Type: {result}")
