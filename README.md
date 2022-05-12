# Golbirev House Classification Project - API Endpoint

Deployed link:
- Heroku: https://golbirev-01.herokuapp.com/predict

This app was made as the final assignment of [TelkomAthon #3 Deep Learning Stream](https://www.telkomathon.com/). 
This repository is using FastAPI as its framework to serve API needed for this project.

The objective of this Machine Learning project is to build a model and web application that the user can use to classify whether the picture uploaded is a valid house picture or not.

This repository **only** serve as API endpoint for the dashboard. The repository for the dashboard (hosted on Heroku with streamlit) can be found here : [Golbirev Streamlit](https://github.com/putawararevalda/golbirev-streamlit)

## Endpoints

1. `/prediction`: Prediction endpoint. Requires body parameters with variable name : `file`
    - Return: 

```json
{
    "prediction_result": {
        "model_output": [0,1], 
        "predicted_class": "NOT_OK" or "OK",
        "confidence_percentage": r[0,100]
    },
    "model_info": {
        "model_name": name_of_model
    },
    "image_info": {
        "image_size": [
            picture_index,
            width_in_pixel,
            height_in_pixel,
            num_of_channels
        ]
    }
}
```

