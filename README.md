# LLM Dynamic Planner (LLM-DP)

## Setup

1. Install ``alfworld`` following instructions [here](https://github.com/alfworld/alfworld).
2. Install requirements: ``pip install -r requirements.txt``.
3. Install docker for running LAPKT planner. There are two different docker images available:
   - The docker image for the linux/arm64 platform is available [here](<https://hub.docker.com/repository/docker/gautierdag/lapkt-arm/general>). See the `Dockerfile` for more details.
   - The docker image for the linux/amd64 platform is available [here](<https://hub.docker.com/r/lapkt/lapkt-public>)

## Config

The config file is a ``.env`` file. The following variables are available:

- ``openai_api_key``: OpenAI API key.
- ``alfworld_data_path``: path to the ``alfworld/data`` directory.
- ``alfworld_config_path``: path to the ``alfworld/configs`` directory.
- ``pddl_dir``: path to the directory where PDDL files are stored.
- ``pddl_domain_file``: name of the PDDL domain file.
- ``planner_solver``: name of the planner solver to use. Currently, only ``bfs_f`` and ``ff`` are supported.
- ``planner_timeout``: timeout for the planner in seconds (default: 30).
- ``planner_cpu_count``: number of CPUs to use for the planner (default: 4).
- ``top_n``: number of plans to generate (default: 3).
- ``platform``: platform to use for the planner (default: ``linux/arm64``).
- ``output_dir``: path to the output directory (default: ``output``).
- ``seed``: random seed (default: 42).
- ``name``: name of the experiment (default: ``llmdp``).
- ``llm_model``: name of the LLM model to use (default: ``gpt-3.5-turbo-0613``).
- ``sample``: whether to use LLM to instantiate beliefs (default: ``llm``).
- ``use_react_chat``: activate ReAct baseline (default: ``False``).
- ``random_fallback``: activate random fallback (default: ``False``).

Copy the ``.env.draft`` to ``.env`` and fill the variables in the created ``.env`` file with the appropriate values.

See `config.py` for more details.

## Run

Run the following command to run the LLM-DP (or ReAct) agent:

```bash
python main.py
```
