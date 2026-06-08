from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_planner: str = "openai:gpt-4o"
    model_researcher: str = "openai:gpt-4o-mini"
    model_synthesizer: str = "openai:gpt-4o"
    model_critic: str = "openai:gpt-4o-mini"

    temperature: float = 0.0
    max_research_iterations: int = 2


settings = Settings()