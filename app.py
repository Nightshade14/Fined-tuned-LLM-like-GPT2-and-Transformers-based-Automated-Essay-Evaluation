from fastapi import FastAPI
from data_schema.Data import Data
from transformers import AutoModelForSequenceClassification

app = FastAPI()

@app.get("/")
def get_home_page():
	return "Welcome to the router!!!"

@app.post("/evaluate")
def evaluate_essay(data: Data):
	data_dict = data.model_dump()
	NUM_CLASSES = 6
	essay = data_dict.get("essay")
	model_id = data_dict.get("model")
	model = AutoModelForSequenceClassification.from_pretrained('./model/my_model', num_labels=NUM_CLASSES)
	return type(data_dict)