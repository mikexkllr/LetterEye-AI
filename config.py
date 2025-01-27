from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings

class Settings(BaseSettings):
    pdf_folder_path: str = Field(default="pdf_folder")
    openai_api_key: str = Field(default="your_openai_api_key")
    csv_files: str = Field(default="csv_files")
    output_dir: str = Field(default="output")
    language: str = Field(default="english")
    unrecognized_dir: str = Field(default="unrecognized")
    failed_dir: str = Field(default="failed")
    unknown_dir: str = Field(default="unknown")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

class Config: 
    settings: Settings = Settings()