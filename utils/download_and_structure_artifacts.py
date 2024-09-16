def get_tokenizers_and_models():
    import os
    import boto3
    import traceback
    from dotenv import load_dotenv

    try:
        # eventually, it will become MODEL_ID_TO_PATH_MAPPING.keys()
        # MODELS_AND_TOKENIZERS_LIST = os.getenv("MODEL_ID_TO_PATH_MAPPING").keys()
        # MODELS_AND_TOKENIZERS_LIST = ["deBERTa-v3", "my_distilbert"]
        MODELS_AND_TOKENIZERS_LIST = ["deBERTa-v3"]

        # Load from a config directory
        load_dotenv(dotenv_path=os.path.join("config", ".env"))

        # Set up AWS credentials (make sure you have the necessary permissions)
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3_client = session.client("s3")

        # Define the S3 bucket and key for the model
        s3_bucket = os.getenv("S3_BUCKET")

        s3_model_dir_path = os.getenv("S3_MODEL_DIR_PATH")
        s3_tokenizer_dir_path = os.getenv("S3_TOKENIZER_DIR_PATH")

        local_model_dir_path = os.getenv("LOCAL_MODEL_DIR_PATH")
        local_tokenizer_dir_path = os.getenv("LOCAL_TOKENIZER_DIR_PATH")

        cwd = os.getcwd()

        file_type = ".zip"
        file_action = "unzip"

        for model_and_tokenizer_name in MODELS_AND_TOKENIZERS_LIST:
            # Check if the model is available locally
            if os.path.exists(f"{cwd}{local_model_dir_path}{model_and_tokenizer_name}"):
                print(f"Model found locally for {model_and_tokenizer_name}.")
            else:
                print(
                    f"Model not found locally for {model_and_tokenizer_name}. Downloading from S3..."
                )
                s3_client.download_file(
                    s3_bucket,
                    f"{s3_model_dir_path}{model_and_tokenizer_name}{file_type}",
                    f"{cwd}{local_model_dir_path}{model_and_tokenizer_name}{file_type}",
                )
                print("Model downloaded successfully.")
                os.system(
                    f"{file_action} {cwd}{local_model_dir_path}{model_and_tokenizer_name}{file_type} -d {cwd}{local_model_dir_path}"
                )
                os.system(
                    f"rm {cwd}{local_model_dir_path}{model_and_tokenizer_name}{file_type}"
                )
                print(f"{model_and_tokenizer_name} model is ready for use.")

            # Check if the tokenizer is available locally
            if os.path.exists(
                f"{cwd}{local_tokenizer_dir_path}{model_and_tokenizer_name}"
            ):
                print(f"Tokenizer found locally for {model_and_tokenizer_name}.")
            else:
                print(
                    f"Tokenizer not found locally for {model_and_tokenizer_name}. Downloading from S3..."
                )
                s3_client.download_file(
                    s3_bucket,
                    f"{s3_tokenizer_dir_path}{model_and_tokenizer_name}{file_type}",
                    f"{cwd}{local_tokenizer_dir_path}{model_and_tokenizer_name}{file_type}",
                )
                print("Tokenizer downloaded successfully.")
                os.system(
                    f"{file_action} {cwd}{local_tokenizer_dir_path}{model_and_tokenizer_name}{file_type} -d {cwd}{local_tokenizer_dir_path}"
                )
                os.system(
                    f"rm {cwd}{local_tokenizer_dir_path}{model_and_tokenizer_name}{file_type}"
                )
                print(f"{model_and_tokenizer_name} tokenizer is ready for use.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
