import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle as pkl
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd

app = FastAPI()

with open(r'C:\Users\Abdullahi Mujaheed\Desktop\mlzoom\mlzoomcamp\first_pipenv\vectorizer_and_model', 'rb') as package:
    vectorizer, model = pkl.load(package)

class InputData(BaseModel):
    timestamp: Optional[str] = None
    lat: float
    lon: float
    severity: str
    cause: str
    vehicles_involved: int
    injuries: int


@app.post('/predict')
def predict(client: InputData | dict):
    # normalize input whether it's a Pydantic model or a plain dict
    if isinstance(client, dict):
        client_dict = client
    else:
        # support pydantic v2 (.model_dump) and v1 (.dict)
        if hasattr(client, "model_dump"):
            client_dict = client.model_dump()
        elif hasattr(client, "dict"):
            client_dict = client.dict()
        else:
            raise ValueError("Unsupported input type")

    # ensure we have a list-of-dicts for vectorizer
    cl = pd.DataFrame([client_dict])

    # compute month from timestamp (accept either 'timestamp' or 'date')
    if 'timestamp' in cl.columns and pd.notna(cl.at[0, 'timestamp']):
        cl['month'] = pd.to_datetime(cl['timestamp']).dt.month
    elif 'date' in cl.columns and pd.notna(cl.at[0, 'date']):
        cl['month'] = pd.to_datetime(cl['date']).dt.month
    else:
        # if month provided directly or missing, keep as-is
        cl['month'] = cl.get('month', pd.Series([None]))

    # vectorize categorical features (pass list of dicts)
    client_dicts = cl[['severity', 'cause']].to_dict(orient='records')
    client_cat = vectorizer.transform(client_dicts)
    vect_data = pd.DataFrame(client_cat, columns=vectorizer.get_feature_names_out())

    # keep numeric columns and join
    r = cl.drop(['severity', 'cause', 'timestamp', 'date'], axis=1, errors='ignore')
    f = r.join(vect_data)

    # make prediction and return JSON-serializable result
    prediction = model.predict(f)
    return {"prediction": int(prediction[0])}