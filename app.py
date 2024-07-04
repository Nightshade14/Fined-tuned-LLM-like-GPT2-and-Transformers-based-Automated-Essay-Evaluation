from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_schema.Data import Data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback

def get_tokenizers_and_models():
	import os
	from transformers import AutoTokenizer, AutoModelForSequenceClassification
	import boto3

	# Set up AWS credentials (make sure you have the necessary permissions)
	session = boto3.Session(
		aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
		aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
	)
	s3_client = session.client('s3')

	# Define the S3 bucket and key for the model
	S3_BUCKET = 'XXXXXXXXXXXXXXXXXXX'
	S3_KEY_PREFIX = 'path/to/distilbert'

	# Define the local path to save the model
	LOCAL_MODEL_PATH = 'path/to/local/model'
	LOCAL_TOKENIZER_PATH = 'path/to/local/tokenizer'

	# Check if the model and tokenizer are available locally
	if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_TOKENIZER_PATH):
		print("Model and tokenizer found locally.")
		tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
		model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
	else:
		print("Model and tokenizer not found locally. Downloading from S3...")
		
		# Download the model and tokenizer from S3
		s3_client.download_file(S3_BUCKET, f"{S3_KEY_PREFIX}/tokenizer/", f"{LOCAL_MODEL_PATH}/tokenizer.json")
		s3_client.download_file(S3_BUCKET, f"{S3_KEY_PREFIX}/config.json", f"{LOCAL_MODEL_PATH}/config.json")
		s3_client.download_file(S3_BUCKET, f"{S3_KEY_PREFIX}/pytorch_model.bin", f"{LOCAL_MODEL_PATH}/pytorch_model.bin")
		
		# Load the model and tokenizer
		tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
		model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
		
	print("Model and tokenizer loaded successfully.")



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
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