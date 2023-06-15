import os
import subprocess
from typing import Literal
import logging

from config import config


def call_lakpt_solver(
    problem_file: str,
    logger: logging.Logger,
    domain_file_path="alfworld_domain.pddl",
    solver: Literal[
        "bfs_f", "dfs_plus", "siw", "siw_plus", "siw-then-bfsf", "ff"
    ] = "bfs_f",
    timeout=5,
) -> list[str]:
    """
    Call the docker LAKPT FF planner and return the plan as list of actions.
    """
    with open("out_problem.pddl", "w") as file:
        file.write(problem_file)

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
        config.platform,  # specify platform in config (.env)
        "--rm",
        "-v",
        f"{os.getcwd()}:/data",
        "lapkt/lapkt-public",
        "timeout",
        f"{timeout}s",
        f"./{solver}",
        "--domain",
        f"/data/{domain_file_path}",
        "--problem",
        "/data/out_problem.pddl",
        "--output",
        "/data/plan.ipc",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.warning(f"Command failed with return code {result.returncode}")
        logger.warning(f"Standard output: {result.stdout.decode()}")
        logger.error(f"Standard error: {result.stderr.decode()}")
        logger.error("Planner Failure")
        return []
    try:
        with open("plan.ipc", "r") as file:
            lines = file.readlines()
            content = [line.strip()[1:-1].lower() for line in lines]
            os.remove("plan.ipc")
            return content
    except FileNotFoundError:
        print("plan.ipc file not found")
