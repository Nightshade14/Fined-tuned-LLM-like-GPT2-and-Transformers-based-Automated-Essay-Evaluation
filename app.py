from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_schema.Data import Data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def get_home_page():
	return "Welcome to the router!!!"

@app.post("/evaluate")
def evaluate_essay(request_data: Data):
	try:

		model_id_to_path_mapping = {
			"DeBERTa-v3": "deBERTa-v3",
			"GPT-2": "my_gpt_2",
			"LSTM": "my_lstm",
			"DistilBERT": "my_distilbert"
		}

		data_dict = request_data.model_dump()
		essay = data_dict.get("essay")
		model_id = data_dict.get("my_model_id")

		TOKENIZER_PATH = "".join(["./models/", model_id_to_path_mapping.get(model_id)])
		MODEL_PATH = "".join(["./models/", model_id_to_path_mapping.get(model_id)]) 
		NUM_CLASSES = 6

		tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
		model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_CLASSES)
		model.eval()

		processed_input = tokenizer(essay, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
		with torch.no_grad():
			outputs = model(**processed_input)
		
		predictions = torch.argmax(outputs.logits, dim=-1).item()

		return {"predicted_class": int(predictions) + 1}

	except Exception as e:
		traceback.print_exc()
		return {"error": str(e)}