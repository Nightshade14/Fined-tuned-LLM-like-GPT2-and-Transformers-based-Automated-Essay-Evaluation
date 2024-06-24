from fastapi import FastAPI
from model.Data import Data

app = FastAPI()

@app.get("/")
def get_home_page():
	return "Welcome to the router!!!"

@app.post("/evaluate")
def evaluate_essay(data: Data):
	data_dict = data.model_dump()
	return 1