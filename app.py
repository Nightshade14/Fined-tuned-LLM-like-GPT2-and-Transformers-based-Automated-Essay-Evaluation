from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback
from dotenv import load_dotenv
import os
from data_schema.Data import Data
from utils.download_and_structure_artifacts import get_tokenizers_and_models

try:
    # Load from a config directory
    load_dotenv(dotenv_path = os.path.join('config', '.env'))

    MODEL_BASE_PATH = os.getenv('LOCAL_MODEL_DIR_PATH')
    TOKENIZER_BASE_PATH = os.getenv('LOCAL_TOKENIZER_DIR_PATH')

    MODEL_ID_TO_PATH_MAPPING = os.getenv('MODEL_ID_TO_PATH_MAPPING')
    print(MODEL_ID_TO_PATH_MAPPING)

    get_tokenizers_and_models()

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
except Exception as e:
    print(e)


@app.get("/")
async def get_home_page():
    return "Welcome to the router!!!"


@app.post("/evaluate")
async def evaluate_essay(request_data: Data):
    try:
        data_dict = request_data.model_dump()
        essay = data_dict.get("essay")
        model_id = data_dict.get("my_model_id")

        # MODEL_ID_TO_PATH_MAPPING = dict(os.getenv('MODEL_ID_TO_PATH_MAPPING'))

        # print(MODEL_ID_TO_PATH_MAPPING)
        MODEL_ID_TO_PATH_MAPPING = {"DeBERTa-v3": "deBERTa-v3","GPT-2": "gpt-2-finetuned","LSTM": "lstm","DistilBERT": "distilBERT"}

        print(TOKENIZER_BASE_PATH)
        print(MODEL_ID_TO_PATH_MAPPING)
        print(model_id)
        print(MODEL_ID_TO_PATH_MAPPING.get(model_id))
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
