from tensorflow.keras.models import load_model

model = load_model("models/eye_model.keras")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale = 1./255)
val_generator = datagen.flow_from_directory(
    "data/train",
    target_size =(64,64),
    color_mode = "grayscale",
    batch_size = 32,
    class_mode = "binary",
    shuffle = "False"
)
loss, accuracy = model.evaluate(val_generator)
print("Overall Accuracy:", accuracy)