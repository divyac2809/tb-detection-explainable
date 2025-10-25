from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.utils import register_keras_serializable

# 1. Register Sequential as a serializable class (fixes deserialization issue)
@register_keras_serializable()
class RegisteredSequential(Sequential):
    pass

# 2. Load model architecture
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# 3. Deserialize the model and load weights
model = model_from_json(loaded_model_json, custom_objects={"Sequential": RegisteredSequential})
model.load_weights("model.h5")

print("✅ Model loaded successfully!")

# 4. Save in new .keras format
model.save("tb_model.keras")
print("✅ Model converted and saved as tb_model.keras")
