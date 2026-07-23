from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    accountAgeDays: int = Field(
        ..., 
        description="Number of days since Account got activated", 
        ge=0
    )
    paymentMethodAgeDays: int = Field(
        ..., 
        description="Number of days since Payment Method got activated", 
        ge=0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "accountAgeDays": 1,
                "paymentMethodAgeDays": 0
            }
        }

class PredictionOutput(BaseModel):
    cluster: int
    segment_name: str