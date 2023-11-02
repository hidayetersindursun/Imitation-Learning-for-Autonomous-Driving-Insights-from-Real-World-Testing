"""
import tensorflow as tf
import resnetModel

model = resnetModel.load_model()
tf.saved_model.save(model,"resnetModel_pb")





"""
"""
import tensorflow as tf

# Load the SavedModel
saved_model_dir = "resnetModel_pb"
saved_model = tf.saved_model.load(saved_model_dir)

# Create a TensorRT converter
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir)

# Optimize the model with TensorRT
converter.convert()

# Save the TensorRT-optimized model
converter.save("resnetModel_optimized_trt")
"""


import tensorflow as tf
import cv2

input_data = cv2.imread("data_collection/data_10_31_14_51/10_31_14_51_22_209.jpg")
input_data = cv2.resize(input_data, (224,224))
input_data=cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB)
input_data.reshape(1,224,224,3)


input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
input_data = tf.expand_dims(input_data, axis=0)  # Add the batch dimension
print(input_data.shape)
# Load the TensorRT-optimized model
optimized_model = tf.saved_model.load("resnetModel_optimized_trt")
print("model loaded")

# Perform inference using the optimized model
output = optimized_model(input_data)
print(output)
