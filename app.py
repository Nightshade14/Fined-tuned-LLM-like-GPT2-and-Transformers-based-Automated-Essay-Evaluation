from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_schema.Data import Data
from transformers import AutoModelForSequenceClassification
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For more security, specify your front-end URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def get_home_page():
	return "Welcome to the router!!!"

@app.post("/evaluate")
def evaluate_essay(data: Data):
	try:
		data_dict = data.model_dump()
		NUM_CLASSES = 6
		essay = data_dict.get("essay")
		model_id = data_dict.get("my_model_id")
		model = AutoModelForSequenceClassification.from_pretrained('./model/my_model', num_labels=NUM_CLASSES)
		return {"my_model_id": model_id}
	except Exception as e:
		traceback.print_exc()
		return {"error": str(e)}