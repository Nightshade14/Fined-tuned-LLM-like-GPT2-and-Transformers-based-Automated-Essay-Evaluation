def get_tokenizers_and_models():
    import os
    import boto3
    import traceback
    from dotenv import load_dotenv

    try:
        load_dotenv()

        # Load from a config directory
        load_dotenv(dotenv_path = os.path.join('config', '.env'))

        # Set up AWS credentials (make sure you have the necessary permissions)
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        s3_client = session.client('s3')

        # Define the S3 bucket and key for the model
        s3_bucket = os.getenv('S3_BUCKET')
        s3_model_prefix = os.getenv('S3_MODEL_PREFIX')
        s3_tokenizer_prefix = os.getenv('S3_TOKENIZER_PREFIX')

        # eventually, it will become MODEL_ID_TO_PATH_MAPPING.keys()
        # MODELS_AND_TOKENIZERS_LIST = os.getenv("MODEL_ID_TO_PATH_MAPPING").keys()
        MODELS_AND_TOKENIZERS_LIST = ["deBERTa-v3"]

        MODEL_BASE_PATH = os.getenv('MODEL_PATH')
        TOKENIZER_BASE_PATH = os.getenv('TOKENIZER_PATH')

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

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
