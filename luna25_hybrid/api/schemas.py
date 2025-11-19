from pydantic import BaseModel

class PredictResponse(BaseModel):
    file: str
    probability: float
    label: int
