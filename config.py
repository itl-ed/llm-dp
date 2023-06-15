from pydantic import BaseSettings


class LLMDPBaseConfig(BaseSettings):
    openai_api_key: str
    alfworld_data_path: str = "alfworld/data"
    alfworld_config_path: str = "alfworld/configs"
    pddl_dir: str = "pddl"
    pddl_domain_file: str = "alfworld_domain.pddl"
    output_dir: str = "output"
    seed: int = 42
    name: str = "search"
    llm_model: str = "text-davinci-003"
    max_steps: int = 50
    platform: str = "linux/amd64"

    class Config:
        env_file = ".env"


LLMDPConfig = LLMDPBaseConfig()
