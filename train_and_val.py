from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load a model
# model = YOLO("yolov8n.yaml")  # Build a new model from scratch
model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training)

"""
# Train and validate the model
model.train(data="config.yaml", epochs=1)
metrics = model.val()  # evaluate model performance on the validation set

# Provide model validation results
metrics.box.maps   # a list contains map50-95 of each category

# Save the model
model.export(format="saved_model")  # export the model to ONNX format
"""

model.tune(data='config.yaml', epochs=1, iterations=10, optimizer='AdamW', plots=False, save=True, val=True)