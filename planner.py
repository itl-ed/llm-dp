import os
import subprocess
from typing import Literal
import logging
from tempfile import NamedTemporaryFile

from config import LLMDPConfig


def call_lakpt_solver(
    problem_file: str,
    logger: logging.Logger,
    solver: Literal[
        "bfs_f", "dfs_plus", "siw", "siw_plus", "siw-then-bfsf", "ff"
    ] = "bfs_f",
    timeout=5,
) -> list[str]:
    """
    Call the docker LAKPT FF planner and return the plan as list of actions.
    """
    problem_temp_file = NamedTemporaryFile(
        mode="w",
        suffix=".pddl",
        prefix="problem_",
        dir=LLMDPConfig.pddl_dir,
        delete=False,
    )
    problem_temp_file.write(problem_file)
    problem_temp_file.close()

    plan_temp_file = NamedTemporaryFile(
        suffix=".ipc",
        prefix="plan_",
        dir=LLMDPConfig.pddl_dir,
        delete=False,
    )
    plan_temp_file.close()

    # example command:
    # docker run --platform linux/amd64 --rm
    # -v ~/gpt-planner:/data lapkt/lapkt-public
    # timeout 5s ./ff --domain /data/alfworld_domain.pddl
    #                 --problem /data/out_problem.pddl
    #                 --output /data/plan.ipc
    command = [
        "docker",
        "run",
        "--platform",
        LLMDPConfig.platform,  # specify platform in config (.env)
        "--rm",
        "-v",
        f"{os.getcwd()}/{LLMDPConfig.pddl_dir}:/data",
        "lapkt/lapkt-public",
        "timeout",
        f"{timeout}s",
        f"./{solver}",
        "--domain",
        f"/data/{LLMDPConfig.pddl_domain_file}",
        "--problem",
        f"/data/{os.path.basename(problem_temp_file.name)}",
        "--output",
        f"/data/{os.path.basename(plan_temp_file.name)}",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.warning(f"Command failed with return code {result.returncode}")
        logger.warning(f"Standard output: {result.stdout.decode()}")
        logger.error(f"Standard error: {result.stderr.decode()}")
        logger.error("Planner Failure")
        return []
    try:
        with open(plan_temp_file.name, "r") as file:
            lines = file.readlines()
            content = [line.strip()[1:-1].lower() for line in lines]
            return content
    except FileNotFoundError:
        logger.warning("{plan_temp_file.name} file not found")
    finally:
        os.remove(plan_temp_file.name)
        os.remove(problem_temp_file.name)
