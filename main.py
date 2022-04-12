from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

#import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize

from keras.preprocessing.image import ImageDataGenerator, smart_resize

model_name_load = "golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3.hdf5"
MODEL = load_model(model_name_load)
CLASS_NAMES = ["NOT_OK","OK"]
GOOD_EXTS = ['jpg', 'png', 'bmp','tiff','jpeg', 'gif']

IMAGE_SIZE = (400,400)
CHANNEL_MODE = "rgb"
SEED_NUMBER = 1610

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#need to install python-multipart

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)

):
    image = read_file_as_image(await file.read())

    # cek nih udah bener belom img batch
    img_batch = np.expand_dims(image, 0)
    img_batch_shape = img_batch.shape

    ds = resize(img_batch, IMAGE_SIZE)/255

    predictions = MODEL.predict(ds)

    probabilities = predictions[0][0]

    if probabilities > 0.5:
        predicted_class = CLASS_NAMES[1]
        confidence = round(probabilities*100, 2) # OK

    else:
        predicted_class = CLASS_NAMES[0]
        confidence = round(probabilities*100, 2) # NOT_OK

    return {
        "prediction_result":{
            "model_output": float(probabilities),
            "predicted_class": predicted_class,
            "confidence": confidence,
        },
        "model_info":{
            "model_name" : model_name_load
        },
        "image_info":{
            "image_size" : img_batch_shape
        }
    }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost', port = 8000)
