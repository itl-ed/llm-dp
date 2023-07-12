from pydantic import BaseSettings
from typing import Literal


class LLMDPBaseConfig(BaseSettings):
    openai_api_key: str
    alfworld_data_path: str = "alfworld/data"
    alfworld_config_path: str = "alfworld/configs"
    pddl_dir: str = "pddl"
    pddl_domain_file: str = "alfworld_domain.pddl"
    planner_solver: Literal[
        "bfs_f", "dfs_plus", "siw", "siw_plus", "siw-then-bfsf", "ff"
    ] = "bfs_f"
    planner_timeout: int = 30
    planner_cpu_count: int = 4
    top_n: int = 3
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/arm64"
    output_dir: str = "output"
    seed: int = 42
    name: str = "random"
    llm_model: str = "text-davinci-003"
    llm_chat_model: str = "gpt-3.5-turbo-0613"
    use_llm_search: bool = False

    class Config:
        env_file = ".env"


LLMDPConfig = LLMDPBaseConfig()
