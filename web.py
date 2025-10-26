

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

# st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    fp = "./model/model.h5"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# X-Ray Classification [Tuberculosis/Normal]

""")


temp = st.file_uploader("Upload X-Ray Image")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")

else:

    img = image.load_img(temp_file.name, target_size=(
        500, 500), color_mode='grayscale')

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/255
    pp_img = np.expand_dims(pp_img, axis=0)

    # predict
    preds = cnn.predict(pp_img)
    if preds >= 0.5:
        out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(
            preds[0][0]))

    else:
        out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(
            1-preds[0][0]))

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)




# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf

# import tempfile
# import os
# from PIL import Image
# from tensorflow.keras.models import load_model, model_from_json
# from gradcam import make_gradcam_heatmap
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# # model = load_model("new_model.keras")
# # -----------------------------
# # 1. PAGE CONFIG
# # -----------------------------
# st.set_page_config(page_title="TB Detection with Explainability", layout="centered")
# st.title("🫁 Tuberculosis Detection System with Explainability")
# st.write("Upload a Chest X-ray image. The system will enhance low-quality images and explain its prediction.")

# # -----------------------------
# # 2. LOAD MODEL
# # -----------------------------
# @st.cache_resource
# def load_model():
#     with open("model/model.json", "r") as f:
#         model_json = f.read()
#     model = model_from_json(model_json)
#     model.load_weights("model/model.h5")
#     return model

# model = load_model()
# st.success("✅ Model loaded successfully!")

# # -----------------------------
# # 3. IMAGE ENHANCEMENT
# # -----------------------------
# # def enhance_image(img_path):
# #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# #     # Apply denoising
# #     img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

# #     # Improve contrast using CLAHE
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     img = clahe.apply(img)

# #     # Resize to model input size
# #     img = cv2.resize(img, (224, 224))
# #     return img

# # -----------------------------
# # 4. GRAD-CAM FUNCTION
# # -----------------------------
# # -----------------------------
# # 4. GRAD-CAM FUNCTIONS
# # -----------------------------
# def get_gradcam_heatmap(model, img_array, layer_name):
#     """
#     Generate Grad-CAM heatmap for the given model and image.
#     """
#     grad_model = tf.keras.models.Model(
#         [model.inputs], 
#         [model.get_layer(layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         loss = predictions[:, pred_index]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
#     return heatmap.numpy()


# def overlay_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
#     """
#     Overlay the heatmap on the original image.
#     """
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     color_map = cv2.applyColorMap(heatmap, colormap)
#     superimposed = cv2.addWeighted(img, 1 - alpha, color_map, alpha, 0)
#     return superimposed

# # -----------------------------
# # 5. FILE UPLOAD SECTION
# # -----------------------------
# uploaded_file = st.file_uploader("📤 Upload a Chest X-ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Save temporary file
#     temp = tempfile.NamedTemporaryFile(delete=False)
#     temp.write(uploaded_file.read())
#     temp.close()

#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
#     st.info("🧩 Enhancing image for better clarity...")

#     # Enhance image
#     enhanced_img = enhance_image(temp.name)
#     st.image(enhanced_img, caption="Enhanced Image", use_container_width=True, clamp=True)

#     # Prepare for prediction
#     img_array = np.expand_dims(enhanced_img, axis=(0, -1))
#     img_array = img_array / 255.0

#     # Predict
#     pred = model.predict(img_array)
#     label = "🦠 Tuberculosis" if pred[0][0] > 0.5 else "✅ Normal"
#     confidence = float(pred[0][0]) if pred[0][0] > 0.5 else 1 - float(pred[0][0])

#     st.subheader("🔍 Prediction Result")
#     st.write(f"**Result:** {label}")
#     st.write(f"**Confidence:** {confidence * 100:.2f}%")
# # -----------------------------
# # 6. EXPLAINABILITY (Grad-CAM)
# # -----------------------------
# st.info("Generating Grad-CAM heatmap... (this may take a few seconds)")

# try:
#     # Generate heatmap from the last convolutional layer
#     heatmap = get_gradcam_heatmap(model, img_array, layer_name='conv2d_4')  # change if different layer
#     overlay = overlay_heatmap(heatmap, cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR))

#     st.image(overlay, caption="🧠 Model Attention Map (Grad-CAM)", use_container_width=True)
#     st.caption("Highlighted regions show where the model focused while making its decision.")
# except Exception as e:
#     st.warning(f"Grad-CAM visualization failed: {e}")

# # -----------------------------
# # 7. FOOTER
# # -----------------------------
# st.markdown("---")
# st.caption("Built with ❤️ by Divya | Explainable TB Detection System | Streamlit + TensorFlow")

