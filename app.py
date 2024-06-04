import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.keras')

def recognize_digit(image):
    if image is not None:
        image = np.array(image)
        image = image / 255.0
        image = image.reshape((1, 28, 28, 1))

        prediction = model.predict(image)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ""

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(num_top_classes=10),
    live=True
)

iface.launch()