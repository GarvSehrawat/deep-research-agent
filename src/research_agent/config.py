from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_planner: str = "openai:gpt-4o"
    model_researcher: str = "openai:gpt-4o-mini"
    model_synthesizer: str = "openai:gpt-4o"
    model_critic: str = "openai:gpt-4o-mini"
    model_judge: str = "openai:gpt-4o-mini"   # cheap model for LLM-as-judge evals

    temperature: float = 0.0
    max_research_iterations: int = 1

    # Eval gates: the agent must stay at/above these mean scores in CI.
    eval_min_faithfulness: float = 0.7
    eval_min_citation_accuracy: float = 0.8
    eval_min_coverage: float = 0.5
    eval_max_cases: int = 0   # 0 = run the whole dataset; >0 caps it (cheap CI runs)


settings = Settings()