# Fined-tuned-LLM-like-GPT2-and-Transformers-based-Automated-Essay-Evaluation

## Motivation
Essays represent a person's critical thinking. It provides a glimpse into a person's mind. This serves as an approach to cultivate thoughts and ideas, and also a way to evaluate a person. This method is used by academia and educational institutions. Essay evaluation has been performed manually by humans; which is subject to many factors and could lead to inconsistency.

## Overview

## Methodolgy & Strategic Design Decisions
1. We chose to load all the models and tokenizers at the start of the server, so we can serve every request within a definitive range of latency.
    - The trade-off is between postponing the download of all the supported tokenizer and models until they are asked for and serving results by all the models with a constant and definitive latency even for the first time.
    - The former approach does not download all the models and tokenizers before initializing the backend Fastapi server. This would only download the tokenizer and the model that is asked for. For e.g., if the LSTM model is not asked for the first 10 days, then it would save local storage space of storing the LSTM model for those 10 days. 
    - But the limitation of this approach is that, for the first request that asks for the new model, it would have to bear the time delay of downlaoding and loading the model. It would also need complex logic of maintaining multiple threads from downloading the new model when it is demanded concurrently.
    - The later approach takes a little bit longer to start the server as it downloads all the tokenizers and models before initializing the backend server. This would also help in serving every request, even the very first request for each model, in a predictive and definitive time without any anomalies.
    - It would also provide the benefit of robust availability and readiness of all the models befores serving the users.

2. Storing the project secrets in a configuration file.
    - The paths and secrets are stored in a configuration instead of storing as environment variables


## Impending work 