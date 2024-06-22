from pydantic import BaseModel
from fastapi import FastAPI

class Data(BaseModel):
	essay: str
	model: str

app = FastAPI()

@app.get("/")
def get_home_page():
	return "Welcome to the router!!!"

@app.post("/evaluate")
def evaluate_essay(data: Data):
	data_dict = data.dict()
	return 1