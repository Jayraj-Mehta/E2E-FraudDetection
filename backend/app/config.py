import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Define variables with their expected types
    APP_NAME: str = "Fraud Detection API"
    DEBUG: bool = True
    
    # Model paths
    MODEL_PATH: str
    PREPROCESSOR_PATH: str

    # Tell Pydantic to look for a .env file
    # case_sensitive=True ensures env keys match exactly
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True
    )

# Instantiate the settings object to be imported across the app
settings = Settings()