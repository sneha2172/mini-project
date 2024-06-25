import tensorflow as tf
import keras as keras
import os

input_shape = (224, 224, 3)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu, input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model_dir = "C:/RLDD"
os.makedirs(model_dir, exist_ok=True)

# Set the paths to the train, val, and test directories
train_dir = "C:/RLDD/dataset/train"
val_dir = "C:/RLDD/dataset/val"
test_dir = "C:/RLDD/dataset/test"

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Create data generators for train, val, and test sets
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Read level indications from text files
def read_level_indications(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        level_indications = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                level_indication = int(parts[1])
                level_indications.append(level_indication)
    return level_indications

train_level_file = "C:/RLDD/dataset/train.txt"
val_level_file = "C:/RLDD/dataset/val.txt"
test_level_file = "C:/RLDD/dataset/test.txt"

train_level_indications = read_level_indications(train_level_file)
val_level_indications = read_level_indications(val_level_file)
test_level_indications = read_level_indications(test_level_file)

# Train the model on the data generators
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the trained model weights to disk
model.save(os.path.join(model_dir, "model_weights.h5"))
