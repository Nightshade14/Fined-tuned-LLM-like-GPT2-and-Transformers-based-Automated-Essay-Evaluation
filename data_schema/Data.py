from pydantic import BaseModel

class Data(BaseModel):
	essay: str
	model: str