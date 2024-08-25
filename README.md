# Fined-tuned-LLM-like-GPT2-and-Transformers-based-Automated-Essay-Evaluation

## Setup
Do "pip install -r requirements.txt" to install all the dependencies.

After configuring or setting env varoiables, simply start the server. The server will only launch if the models and tokenizers are present on AWS S3 storage. They are stored as zip files to reduce file size. The artifacts would be downlaoded, extracted (unziped), and the zip files would be deleted. After all these processes are succcessful, only then the FastAPI server will start serving requests. 

Note: For experimenting with this project, you will need to store the models and tokenizers on your AWS S3 storage and you will need your own AWS credentials. The model and tokenizer's public access is blocked.

Note: The Nginx files were for local use. The frontend was served with Nginx acting as a web server for frontend.

## Motivation
Essays represent a person's critical thinking. It provides a glimpse into a person's mind. This serves as an approach to cultivate thoughts and ideas, and also a way to evaluate a person. This method is used by academia and educational institutions. Essay evaluation has been performed manually by humans; which is subject to many factors and could lead to inconsistency.

## Working
The user provides their essay on a web-form and also selects the AI model that will evaluate the essay. The request would be forwarded to backend fastapi server and it will evaluate the essay and send the score back to the user, below the form.

## Strategic Design Decisions
1. We chose to load all the models and tokenizers at the start of the server, so we can serve every request within a definitive range of latency.
    - The trade-off is between postponing the download of all the supported tokenizer and models until they are asked for and serving results by all the models with a constant and definitive latency even for the first time.
    - The former approach does not download all the models and tokenizers before initializing the backend Fastapi server. This would only download the tokenizer and the model that is asked for. For e.g., if the LSTM model is not asked for the first 10 days, then it would save local storage space of storing the LSTM model for those 10 days. 
    - But the limitation of this approach is that, for the first request that asks for the new model, it would have to bear the time delay of downlaoding and loading the model. It would also need complex logic of maintaining multiple threads from downloading the new model when it is demanded concurrently.
    - The later approach takes a little bit longer to start the server as it downloads all the tokenizers and models before initializing the backend server. This would also help in serving every request, even the very first request for each model, in a predictive and definitive time without any anomalies.
    - It would also provide the benefit of robust availability and readiness of all the models befores serving the users.

2. Storing the project secrets in a configuration file.
    - The paths and secrets are stored in a configuration instead of storing as environment variables. Both methods provide security to the sensitive data and credentials. For local development, environment variables of multiple projects can overlap. But for production environment, the secrets must be stored as environment variables or they can be saved with AWS Secrets Manager for decentralized security and flexibility and freedom of configuring an environment.


## Next-step:
1. Try out Llama 3.1 8B for this use case. Its 128k context window will support large essays.


## Impending work:
- Convert Model to ONNX and serve with TensorRT to decrease inference latency by around 2-3x
- Hosting the work live on AWS EC2 or on GitHub Pages with CI/CD with GitHub Actions.
- Improving the Quadratic Kappa Score of the models, combat overfitting and acquire more essay data so the large models have sufficient data to train on.


