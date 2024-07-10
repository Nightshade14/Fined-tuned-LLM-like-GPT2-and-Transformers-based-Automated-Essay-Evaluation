from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_schema.Data import Data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback
from dotenv import load_dotenv
import os

# Load from the root directory
load_dotenv()

# Load from a config directory
load_dotenv(dotenv_path=os.path.join('config', '.env'))

MODEL_BASE_PATH = os.getenv('MODEL_PATH')
TOKENIZER_BASE_PATH = os.getenv('TOKENIZER_PATH')

MODEL_ID_TO_PATH_MAPPING = {
    "DeBERTa-v3": "deBERTa-v3",
    "GPT-2": "my_gpt_2",
    "LSTM": "my_lstm",
    "DistilBERT": "my_distilbert"
}

MODELS_AND_TOKENIZERS_LIST = ["deBERTa-v3"]


def get_tokenizers_and_models():
    import os
    import boto3

    # Set up AWS credentials (make sure you have the necessary permissions)
    session = boto3.Session(
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    s3_client = session.client('s3')

    # Define the S3 bucket and key for the model
    s3_bucket = 'XXXXXXXXXXXXXXXXXXX'
    s3_model_prefix = './models/'
    s3_tokenizer_prefix = './tokenizers/'

    # Define the local path to save the model
    local_model_path = MODEL_BASE_PATH
    local_tokenizer_path = TOKENIZER_BASE_PATH

    for model_and_tokenizer in MODELS_AND_TOKENIZERS_LIST:
        # Check if the model and tokenizer are available locally
        if os.path.exists(f"{local_model_path}{model_and_tokenizer}") and os.path.exists(
            f"{local_tokenizer_path}{model_and_tokenizer}"):
            print(f"Model and tokenizer found locally for {model_and_tokenizer}.")
        elif not os.path.exists(f"{local_model_path}{model_and_tokenizer}"):
            print(f"Model not found locally for {model_and_tokenizer}. Downloading from S3...")

            s3_client.download_file(s3_bucket, f"{s3_model_prefix}{model_and_tokenizer}.zip",
                                    f"{local_model_path}{model_and_tokenizer}.zip")
            os.system(f"unzip {local_model_path}{model_and_tokenizer}.zip")
            os.system(f"rm {local_model_path}{model_and_tokenizer}.zip")
        elif not os.path.exists(f"{local_tokenizer_path}{model_and_tokenizer}"):
            print(f"Tokenizer not found locally for {model_and_tokenizer}. Downloading from S3...")

            s3_client.download_file(s3_bucket, f"{s3_tokenizer_prefix}{model_and_tokenizer}.zip",
                                    f"{local_tokenizer_path}{model_and_tokenizer}.zip")
            os.system(f"unzip {local_tokenizer_path}{model_and_tokenizer}.zip")
            os.system(f"rm {local_tokenizer_path}{model_and_tokenizer}.zip")
        else:
            print(f"Model and tokenizer not found locally for {model_and_tokenizer}. Downloading from S3...")

            s3_client.download_file(s3_bucket, f"{s3_model_prefix}{model_and_tokenizer}.zip",
                                    f"{local_model_path}{model_and_tokenizer}.zip")
            s3_client.download_file(s3_bucket, f"{s3_tokenizer_prefix}{model_and_tokenizer}.zip",
                                    f"{local_tokenizer_path}{model_and_tokenizer}.zip")

            os.system(f"unzip {local_model_path}{model_and_tokenizer}.zip")
            os.system(f"unzip {local_tokenizer_path}{model_and_tokenizer}.zip")

            os.system(f"rm {local_model_path}{model_and_tokenizer}.zip")
            os.system(f"rm {local_tokenizer_path}{model_and_tokenizer}.zip")


get_tokenizers_and_models()

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
        data_dict = request_data.model_dump()
        essay = data_dict.get("essay")
        model_id = data_dict.get("my_model_id")

        tokenizer_path = "".join([TOKENIZER_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get(model_id)])
        model_path = "".join([MODEL_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get(model_id)])
        num_classes = 6

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
        model.eval()

        processed_input = tokenizer(essay, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**processed_input)

        predictions = torch.argmax(outputs.logits, dim=-1).item()

        return {"predicted_class": int(predictions) + 1}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
