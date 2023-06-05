import subprocess
import requests
import os


def call_downward_solver(
    problem_file: str, domain_file_path="domain_v5.pddl"
) -> list[str]:
    """
    Call the local Downward planner and return the plan as list of actions.
    """
    with open("out_problem.pddl", "w") as file:
        file.write(problem_file)
    # run the external command for FF planner
    command = [
        "python",
        "downward/fast-downward.py",
        domain_file_path,
        "out_problem.pddl",
        "--evaluator",
        "hff=ff()",
        "--search",
        "lazy_greedy([hff], preferred=[hff])",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"Standard error: {result.stderr.decode()}")

    # read the results from the 'sas_plan' file
    try:
        with open("sas_plan", "r") as file:
            lines = file.readlines()
            content = [line.strip()[1:-1] for line in lines[:-1]]
        # remove the temporary files
        os.remove("sas_plan")

        return content
    except FileNotFoundError:
        print("sas_plan file not found")


def call_planning_domains_solver(
    problem: str, domain_file_path="domain_v5.pddl"
) -> list[str]:
    # run remote PDDL planner
    data = {"domain": open(domain_file_path, "r").read(), "problem": problem}
    resp = requests.post(
        "http://solver.planning.domains/solve", verify=False, json=data
    ).json()
    if "result" not in resp:
        print("No plan found.")
        print(resp)
        return None
    try:
        plan = resp["result"]["plan"]
        plan = [p["name"][1:-1] for p in plan]
        return plan
    except KeyError:
        print(resp)
