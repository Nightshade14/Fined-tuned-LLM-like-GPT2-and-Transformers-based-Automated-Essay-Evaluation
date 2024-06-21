from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def get_home_page():
	return "Welcome to the home page!!!"

@app.get("/models/{model_id}")
def return_selected_model_id(model_id):
	return f"{model_id}"
