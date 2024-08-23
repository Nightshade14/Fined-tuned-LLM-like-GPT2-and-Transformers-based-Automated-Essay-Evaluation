from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
import os
import json


env_path = os.path.join("..", "config", ".env")
load_dotenv(dotenv_path = env_path)


MODEL_ID_TO_PATH_MAPPING = json.loads(os.getenv('MODEL_ID_TO_PATH_MAPPING'))
MODEL_BASE_PATH = os.getenv('LOCAL_MODEL_DIR_PATH')
TOKENIZER_BASE_PATH = os.getenv('LOCAL_TOKENIZER_DIR_PATH')

tokenizer_path = "".join([".", TOKENIZER_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get("DeBERTa-v3")])
model_path = "".join([".", MODEL_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get("DeBERTa-v3")])
num_classes = 6

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("tokenizer loaded")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
print("model loaded\n")

model.eval()
print("model in eval mode")

onnx_model_path = "../artifacts/onnx/deberta_v3.onnx"
dummy_input = tokenizer("This is a dummy input", return_tensors="pt", padding="max_length", truncation=True, max_length=512)
print("Dummy input generated.\nStarting model export to ONNX...")
torch.onnx.export(model, 
                    (dummy_input['input_ids'], dummy_input['attention_mask']), 
                    onnx_model_path, 
                    input_names=['input_ids', 'attention_mask'], 
                    output_names=['logits'], 
                    dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'logits': {0: 'batch_size'}})

print("Pytorch Model successfully exported to ONNX model")