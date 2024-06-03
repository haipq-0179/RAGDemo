from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TracingConfig(BaseModel):
    public_key: str
    secret_key: str
    user_id: str
    host: str = "https://cloud.langfuse.com"
    flush_at: int = 2


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    key: str
    gpt_deployment_name: str
    embed_deployment_name: str
    version: str


class AzureSpeechConfig(BaseModel):
    endpoint: str
    region: str
    key: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    azure_openai: AzureOpenAIConfig
    azure_speech: AzureSpeechConfig
    tracing: TracingConfig


def load_settings() -> Settings:
    return Settings()