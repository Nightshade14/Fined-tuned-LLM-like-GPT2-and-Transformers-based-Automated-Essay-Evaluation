from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import traceback
from dotenv import load_dotenv
import os
from data_schema.Data import Data
from utils.download_and_structure_artifacts import get_tokenizers_and_models
import json
import time
import onnxruntime as ort
import numpy as np


try:
    app = FastAPI()
    # app.add_middleware(CProfileMiddleware,
    #     enable=True,
    #     print_each_request = True,
    #     strip_dirs = False,
    #     sort_by='cumulative'
    # )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load from a config directory
    load_dotenv(dotenv_path=os.path.join("config", ".env"))

    MODEL_BASE_PATH = os.getenv("LOCAL_MODEL_DIR_PATH")
    TOKENIZER_BASE_PATH = os.getenv("LOCAL_TOKENIZER_DIR_PATH")

    MODEL_ID_TO_PATH_MAPPING = json.loads(os.getenv("MODEL_ID_TO_PATH_MAPPING"))

    get_tokenizers_and_models()

    tokenizer_path = "".join(
        [TOKENIZER_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get("DeBERTa-v3")]
    )
    model_path = "".join([MODEL_BASE_PATH, MODEL_ID_TO_PATH_MAPPING.get("DeBERTa-v3")])
    num_classes = 6

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


except Exception as e:
    print(e)


@app.get("/")
async def get_home_page():
    return "Welcome to the router!!!"


@app.post("/evaluate")
async def evaluate_essay(request_data: Data):
    try:
        time1 = time.time()

        data_dict = request_data.model_dump()
        essay = data_dict.get("essay")
        model_id = data_dict.get("my_model_id")

        processed_input = tokenizer(
            essay,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        session = ort.InferenceSession("".join([model_path, ".onnx"]))

        # Get model input and output names
        input_names = [input.name for input in session.get_inputs()]
        output_name = session.get_outputs()[0].name

        input_feed = {
            input_names[0]: processed_input["input_ids"].astype(np.int64),
            input_names[1]: processed_input["attention_mask"].astype(np.int64),
        }

        outputs = session.run([output_name], input_feed)
        print(outputs)
        time2 = time.time()
        print(f"Time Taken to serve request: {time2 - time1}")
        predicted_class = np.argmax(outputs[0], axis=-1)

        return {"predicted_class": int(predicted_class)}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
