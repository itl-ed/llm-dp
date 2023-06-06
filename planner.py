import subprocess
import requests
import os


def call_downward_solver(
    problem_file: str, domain_file_path="domain_v5.pddl"
) -> list[str]:
    """
    Call the local FF planner and return the plan as list of actions.
    """
    with open("out_problem.pddl", "w") as file:
        file.write(problem_file)
    # run the external command for FF planner
    # command = [
    #     "python",
    #     "downward/fast-downward.py",
    #     domain_file_path,
    #     "out_problem.pddl",
    #     "--evaluator",
    #     "hff=ff()",
    #     "--search",
    #     "lazy_greedy([hff], preferred=[hff])",
    # ]

    # python downward/fast-downward.py --alias seq-sat-lama-2011 --overall-time-limit 2s domain_v5.pddl out_problem.pddl
    # TODO - if error 21 - timeout --> try to increase the timeout
    command = [
        "python",
        "downward/fast-downward.py",
        "--alias",
        "seq-sat-lama-2011",
        "--overall-time-limit",
        "2s",
        domain_file_path,
        "out_problem.pddl",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return code 23 means timeout
    if result.returncode != 0 and result.returncode != 23:
        print(f"Command failed with return code {result.returncode}")
        print(f"Standard error: {result.stderr.decode()}")
        raise Exception(f"Plan Fail {result.returncode}: {result.stderr.decode()}")

    # read the results from the 'sas_plan' file
    try:
        # list sas_plan_files (sas_plan.1, sas_plan.2, ...)
        sas_plan_files = [
            f for f in os.listdir(".") if os.path.isfile(f) and f.startswith("sas_plan")
        ]
        # sort and select the last sas_plan_file
        sas_plan_files.sort()
        sas_plan_file = sas_plan_files[-1]
        with open(sas_plan_file, "r") as file:
            lines = file.readlines()
            content = [line.strip()[1:-1] for line in lines[:-1]]

        # clean all sas_plan_files
        for sas_plan_file in sas_plan_files:
            os.remove(sas_plan_file)

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
