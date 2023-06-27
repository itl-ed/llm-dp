import os
import subprocess
import multiprocessing
from functools import partial
import logging
from tempfile import NamedTemporaryFile

from config import LLMDPConfig


def call_lakpt_solver(
    problem_file: str,
    logger: logging.Logger,
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
    # timeout 30s ./ff --domain /data/alfworld_domain.pddl
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
        f"{LLMDPConfig.planner_timeout}s",
        f"./{LLMDPConfig.planner_solver}",
        "--domain",
        f"/data/{LLMDPConfig.pddl_domain_file}",
        "--problem",
        f"/data/{os.path.basename(problem_temp_file.name)}",
        "--output",
        f"/data/{os.path.basename(plan_temp_file.name)}",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logger.warning(f"Command failed with return code {result.returncode}")
            logger.warning(f"Standard output: {result.stdout.decode()}")
            logger.error(f"Standard error: {result.stderr.decode()}")
            logger.error("Planner Failure")
            return []
        with open(plan_temp_file.name, "r") as file:
            lines = file.readlines()
            content = [
                line.strip()[1:-1].lower() for line in lines if "REACH-GOAL" not in line
            ]
            return content
    except FileNotFoundError:
        logger.warning("{plan_temp_file.name} file not found")
    finally:
        os.remove(plan_temp_file.name)
        os.remove(problem_temp_file.name)


def parallel_lakpt_solver(
    problems: list[str], logger: logging.Logger
) -> list[list[str]]:
    with multiprocessing.Pool(processes=LLMDPConfig.planner_cpu_count) as pool:
        # Map the process_file function to the list of files
        results = pool.map(partial(call_lakpt_solver, logger=logger), problems)
    # filter out empty plans
    results = [result for result in results if result]
    return results
