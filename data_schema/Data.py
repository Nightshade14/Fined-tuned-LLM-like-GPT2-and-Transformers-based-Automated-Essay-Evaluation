from pydantic import BaseModel

class Data(BaseModel):
	essay: str
	my_model_id: str