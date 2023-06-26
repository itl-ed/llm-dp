import json
import os
import pickle
import random
import re
import time
import logging
from collections import defaultdict, Counter

import openai
import tiktoken
import yaml
from openai.error import RateLimitError

from logger import get_logger
from config import LLMDPConfig
from planner import call_lakpt_solver

# Set seed for reproducibility
random.seed(LLMDPConfig.seed)

encoding = tiktoken.encoding_for_model(LLMDPConfig.llm_model)
chat_encoding = tiktoken.encoding_for_model(LLMDPConfig.llm_chat_model)
openai.api_key = LLMDPConfig.openai_api_key


GENERATE_GOAL_PROMPT = """(define (domain alfred)
(:predicates
(isReceptacle ?o - object) ; true if the object is a receptacle
(atReceptacleLocation ?r - object) ; true if the robot is at the receptacle location
(inReceptacle ?o - object ?r - object) ; true if object ?o is in receptacle ?r
(openable ?r - object) ; true if a receptacle is openable
(opened ?r - object) ; true if a receptacle is opened
(isLight ?o - object) ; true if an object is light source
(examined ?o - object ?l - object) ; whether the object has been looked at with light
(holds ?o - object) ; object ?o is held by robot
(isClean ?o - object) ; true if the object has been cleaned in sink
(isHot ?o - object) ; true if the object has been heated up
(isCool ?o - object) ; true if the object has been cooled
(isSink ?o - object) ; true if the object is a sink
(isMicrowave ?o - object) ; true if the object is a microwave
(isFridge ?o - object) ; true if the object is a fridge
)
)
TASK: Your task is to: put a clean plate in microwave.
(:goal
(exists (?t - plate ?r - microwave)
(and (inReceptacle ?t ?r)
(isClean ?t)
)))
TASK: Your task is to: examine an alarmclock with the desklamp
(:goal
(exists (?t - alarmclock ?l - desklamp)
(and (examined ?t ?l) (holds ?t)
)))
TASK: Your task is to: put two cellphone in bed
(:goal
(exists (?t1 - cellphone ?t2 - cellphone ?r - bed)
(and (inReceptacle ?t1 ?r)
(inReceptacle ?t2 ?r)
(not (= ?t1 ?t2))
)))"""


def llm(prompt: str, stop=["\n"]) -> str:
    try:
        response = openai.Completion.create(
            model=LLMDPConfig.llm_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return response["choices"][0]["text"]
    except RateLimitError:
        time.sleep(10)
        return llm(prompt, stop=stop)


def llm_cache(prompt: str, stop=["\n"]) -> str:
    """
    cache same llm responses into a pickle file
    This avoids paying for the same prompt multiple times
    It also allows us to replicate the results
    """
    cache_file = (
        f"{LLMDPConfig.output_dir}/llm_responses_{LLMDPConfig.llm_model}.pickle"
    )
    llm_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_responses = pickle.load(f)
    if prompt not in llm_responses:
        llm_responses[prompt] = llm(prompt, stop=stop)
        with open(cache_file, "wb") as f:
            pickle.dump(llm_responses, f)
    return llm_responses[prompt]


def llm_find_objects(
    objects: list[str],
    receptacles: list[str],
    logger: logging.Logger,
) -> tuple[str, int]:
    """
    Uses the new function cabability of LLM to find most likely location of objects
    """
    cache_file = (
        f"{LLMDPConfig.output_dir}/llm_responses_{LLMDPConfig.llm_chat_model}.pickle"
    )
    llm_functions = [
        {
            "name": "find_object_at_location",
            "description": "Pass most likely location where the items can be found",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": receptacles,
                        "description": "Most likely location",
                    }
                },
                "required": ["location"],
            },
        }
    ]
    example_user_input = "Find the following objects: " + ", ".join(objects)
    llm_messages = [{"role": "user", "content": example_user_input}]
    num_tokens = len(chat_encoding.encode(str(llm_messages) + str(llm_functions)))

    llm_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_responses = pickle.load(f)
    key = str(objects) + str(receptacles)

    if key not in llm_responses:
        try:
            completion = openai.ChatCompletion.create(
                model=LLMDPConfig.llm_chat_model,
                messages=llm_messages,
                functions=llm_functions,
                function_call={"name": "find_object_at_location"},
                temperature=0.0,
            )
        except RateLimitError:
            time.sleep(10)
            return llm_find_objects(objects, receptacles, logger=logger)

        reply_content = completion.choices[0].message
        funcs = reply_content.to_dict()["function_call"]["arguments"]

        llm_responses[key] = json.loads(funcs)["location"]
        with open(cache_file, "wb") as f:
            pickle.dump(llm_responses, f)

    if llm_responses[key] not in receptacles:
        # hallucinated response
        logger.warning(
            f"Hallucinated response looking for {str(objects)}: {llm_responses[key]}"
        )
        return random.choice(receptacles), num_tokens

    return llm_responses[key], num_tokens


class LLMDPAgent:
    """
    Alfworld agent that uses the LLMDP planner to generate a plan to complete a task.
    """

    def __init__(
        self,
        initial_scene_observation: str,
        task_description: str,
        env_id=0,
        logger=None,
        use_llm_search=False,
        top_n=3,
    ) -> None:
        self.initial_scene_observation = initial_scene_observation
        self.task_description = task_description
        self.env_id = env_id
        self.llm_tokens_sent = 0
        self.llm_chat_tokens_sent = 0
        self.logger = logger
        self.use_llm_search = use_llm_search
        self.top_n = top_n

        # get PDDL objects from scene_observation
        self.scene_receptacles = self.find_receptacles_from_scene_observation(
            initial_scene_observation
        )

        # populate scene_objects with all objects/receptacles in scene
        self.scene_objects = defaultdict(dict)
        for r in self.scene_receptacles:
            self.scene_objects[r] = self.get_receptacle_attributes(r)

        self.pddl_goal = self.get_pddl_goal()
        self.logger.info(f"PDDL GOAL: {self.pddl_goal}")

        # Get objects that should exist based on generated PDDL goal
        existential_pattern = r"\?\w+\s+-\s+(\w+)"
        for o, count in Counter(
            re.findall(existential_pattern, self.pddl_goal)
        ).items():
            for i in range(1, int(count) + 1):
                name = f"{o}-{i}"
                self.scene_objects[name]["type"] = o
                self.scene_objects[name]["seen"] = False
                if "lamp" in name:
                    self.scene_objects[name]["isLight"] = True

        self.plan = self.get_plan()

        self.actions = []

    def llm(self, prompt, stop=["\n"]):
        self.llm_tokens_sent += len(encoding.encode(prompt))
        return llm_cache(prompt, stop)

    @staticmethod
    def get_object_count_id(object_list: list[str]) -> list[str]:
        """
        Given a list of objects, return a list of objects with a count id appended to each object.
        """
        count_dict = {}
        output_list = []

        for item in object_list:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
            output_list.append(f"{item}-{count_dict[item]}")

        return output_list

    def get_task_goal(
        self,
        task_description: str,
    ) -> list[str]:
        """
        Given a task description, relevant objects/receptacles return the PDDL goal.
        """
        prompt = f"{GENERATE_GOAL_PROMPT}\nTASK: {task_description}\n(goal:"
        pddl_goal = self.llm(prompt, stop=None).strip()
        return pddl_goal

    @staticmethod
    def process_obs(observation: str) -> dict:
        """
        No LLM version of process_obs prefered for $ efficiency.
        """
        json_dict = {}
        # check if the receptacle is closed
        closed_receptacle = re.search(r"The (\w+ \d+) is closed", observation)
        if closed_receptacle:
            return json_dict
        # find the receptacle
        receptacle = re.search(
            r"(On|In) the (\w+ \d+)|You open the (\w+ \d+)", observation
        )
        if receptacle:
            # get the receptacle from the right group
            receptacle_key = (
                receptacle.group(2) if receptacle.group(2) else receptacle.group(3)
            )
            receptacle_key = receptacle_key.replace(" ", "-")
            json_dict[receptacle_key] = []
            # check if there's nothing in the receptacle
            no_items = re.search(r"you see nothing", observation)
            if no_items:
                return json_dict
            # find items in the receptacle
            items = re.findall(r"a (\w+ \d+)", observation)
            for item in items:
                json_dict[receptacle_key].append(item.replace(" ", "-"))
        return json_dict

    @staticmethod
    def find_receptacles_from_scene_observation(scene_observation: str) -> list[str]:
        """
        Given an Alfworld initial scene observation, return a list of receptacles.
        """
        receptacles = re.findall(r"a (\w+ \d+)", scene_observation)
        receptacles = [recep.replace(" ", "-") for recep in receptacles]
        return receptacles

    @staticmethod
    def get_receptacle_attributes(receptacle: str) -> dict:
        """
        Given a receptacle, return a dictionary of attributes.
        """
        attributes = {}

        # predicate special types
        if "sink" in receptacle:
            attributes["isSink"] = True
        elif "microwave" in receptacle:
            attributes["isMicrowave"] = True
        elif "fridge" in receptacle:
            attributes["isFridge"] = True

        # predicate openable
        if (
            "microwave" in receptacle
            or "fridge" in receptacle
            or "drawer" in receptacle
            or "cabinet" in receptacle
            or "safe" in receptacle
        ):
            attributes["openable"] = True

        attributes["type"] = receptacle.split("-")[0]
        attributes["seen"] = False
        attributes["isReceptacle"] = True

        return attributes

    @staticmethod
    def convert_pddl_action_to_alfworld(
        action_name: str, action_args: list[str]
    ) -> str:
        """
        Given a PDDL action, convert it to an Alfworld textworld action.
        """
        match action_name:
            case "examineobjectinlight":
                out = f"use {action_args[1]}"
            case "gotoreceptacle":
                out = f"go to {action_args[0]}"
            case "openreceptacle":
                out = f"open {action_args[0]}"
            case "closereceptacle":
                out = f"close {action_args[0]}"
            case "pickupobjectfromreceptacle":
                out = f"take {action_args[0]} from {action_args[1]}"
            case "putobject":
                out = f"put {action_args[0]} in/on {action_args[1]}"
            case "cleanobject":
                out = f"clean {action_args[0]} with {action_args[1]}"
            case "heatobject":
                out = f"heat {action_args[0]} with {action_args[1]}"
            case "coolobject":
                out = f"cool {action_args[0]} with {action_args[1]}"
            case _:
                raise ValueError(f"Unknown action: {action_name}")
        return out.replace("-", " ")

    def get_pddl_goal(self) -> str:
        # use LLM to complete pddl goal
        completed_pddl_goal = self.get_task_goal(self.task_description)
        return f"(:goal {completed_pddl_goal}"

    def get_pddl_objects(self) -> str:
        # get all objects/receptacles in scene
        objects_str = "".join(
            [f"{o} - {atts['type']}\n" for o, atts in self.scene_objects.items()]
        )
        return f"(:objects {objects_str})\n"

    def get_pddl_init(self) -> list[str]:
        # fill in known predicates from observation
        fixed_predicates = ""
        # fixed predicates
        for r, atts in self.scene_objects.items():
            for att, val in atts.items():
                if att in ["type", "seen"] or val is False:
                    continue
                if val is True:
                    fixed_predicates += f"({att} {r})\n"
                else:
                    fixed_predicates += f"({att} {r} {val})\n"

        # dynamic predicates (World Beliefs)
        unseen_receptacles = [
            r
            for r, atts in self.scene_objects.items()
            if not atts["seen"] and "isReceptacle" in atts
        ]
        unseen_objects = [
            o
            for o, atts in self.scene_objects.items()
            if not atts["seen"] and "isReceptacle" not in atts
        ]

        # if no unseen objects, return fixed predicates
        if len(unseen_objects) == 0:
            return [f"(:init {fixed_predicates})\n"]

        belief_predicates = []
        for _ in range(self.top_n):
            # if no unseen receptacles
            if len(unseen_receptacles) == 0:
                break

            belief_predicate = fixed_predicates
            # Use LLM to guess which receptacle
            if self.use_llm_search:
                likely_receptacle, llm_tokens = llm_find_objects(
                    unseen_objects, unseen_receptacles, self.logger
                )
                self.llm_chat_tokens_sent += llm_tokens
            # Use a random unseen receptacle otherwise
            else:
                likely_receptacle = random.choice(unseen_receptacles)

            # remove likely receptacle from unseen receptacles
            unseen_receptacles.remove(likely_receptacle)

            # guess that object in likely receptacle
            for o in unseen_objects:
                belief_predicate += f"(inReceptacle {o} {likely_receptacle})\n"

            belief_predicates.append(belief_predicate)

        return [f"(:init {predicates})\n" for predicates in belief_predicates]

    def get_pddl_problem(self) -> list[str]:
        # get n different init configurations
        inits = self.get_pddl_init()

        problems = []
        for init in inits:
            # construct to PDDL problem.pddl
            problems.append(
                "(define (problem alf)\n(:domain alfred)\n"
                + f"{self.get_pddl_objects()}{init}{self.pddl_goal})"
            )
        return problems

    def update_observation(self, observation: str) -> bool:
        if observation == "":
            return False
        scene_obs = {}
        scene_changed = True

        # use last action to update scene_objects
        action_args = self.actions[-1]
        action_name = action_args[0]
        action_args = action_args[1:]

        # we use the last action to update the scene
        # NOTE: this is using the symbolic :effects of the action
        #       as described in the PDDL domain
        match action_name:
            case "examineobjectinlight":
                self.scene_objects[action_args[0]]["examined"] = action_args[1]
            case "gotoreceptacle":
                for receptacle in self.scene_objects:
                    self.scene_objects[receptacle]["atReceptacleLocation"] = False
                self.scene_objects[action_args[0]]["atReceptacleLocation"] = True
                scene_obs = self.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "openreceptacle":
                self.scene_objects[action_args[0]]["opened"] = True
                scene_obs = self.process_obs(observation)
                scene_changed = len(scene_obs) > 0
            case "closereceptacle":
                self.scene_objects[action_args[0]]["opened"] = False
            case "pickupobjectfromreceptacle":
                del self.scene_objects[action_args[0]]["inReceptacle"]
                self.scene_objects[action_args[0]]["holds"] = True
            case "putobject":
                self.scene_objects[action_args[0]]["holds"] = False
                self.scene_objects[action_args[0]]["inReceptacle"] = action_args[1]
            case "coolobject":
                self.scene_objects[action_args[0]]["isCool"] = True
                self.scene_objects[action_args[0]]["isHot"] = False
            case "heatobject":
                self.scene_objects[action_args[0]]["isHot"] = True
                self.scene_objects[action_args[0]]["isCool"] = False
            case "cleanobject":
                self.scene_objects[action_args[0]]["isClean"] = True

        # use observation to update scene_objects
        for receptacle, objects in scene_obs.items():
            self.scene_objects[receptacle]["seen"] = True

            # if you can see objects in receptacle, it must be opened
            if "openable" in self.scene_objects[receptacle]:
                self.scene_objects[receptacle]["opened"] = True

            # remove inReceptacle from all objects not observed at this receptacle
            for obj in self.scene_objects.keys():
                if (
                    "inReceptacle" in self.scene_objects[obj]
                    and self.scene_objects[obj]["inReceptacle"] == receptacle
                ):
                    del self.scene_objects[obj]["inReceptacle"]

            # update inReceptacle for all objects observed at this receptacle
            # TODO: Test without: NOTE: we ignore seen objects that are not relevant to the task
            # We also add objects of the same type as the target object
            for obj in objects:
                if obj in self.scene_objects:
                    self.scene_objects[obj]["type"] = obj.split("-")[0]
                    self.scene_objects[obj]["seen"] = True
                    self.scene_objects[obj]["inReceptacle"] = receptacle

        return scene_changed

    def get_plan(self) -> list[str]:
        problems = self.get_pddl_problem()
        plans = []
        for problem in problems:
            plan = call_lakpt_solver(problem, solver="bfs_f", logger=self.logger)
            # validate plan
            if len(plan) > 0:
                plans.append(plan)

        # greedy selection strategy
        return min(plans, key=len)

    def take_action(self, last_observation="") -> str:
        # sometimes move action doesn't trigger observation (e.g. if already close)
        if (
            last_observation == "Nothing happens."
            and self.actions[-1][0] == "gotoreceptacle"
        ):
            return f"examine {self.actions[-1][1]}".replace("-", " ")
        # if last action was not move, then we should have observed something
        elif last_observation == "Nothing happens.":
            raise Exception(f"Action {self.actions[-1][0]} failed.")

        # update scene_objects with last observation
        changed = self.update_observation(last_observation)

        # if env changed, replan
        if changed:
            self.plan = self.get_plan()

        # get next action from plan
        pddl_action = self.plan.pop(0)
        # remove parentheses
        action_args = pddl_action.split(" ")

        # append action args to list of actions taken
        self.actions.append(action_args)

        alfworld_action = self.convert_pddl_action_to_alfworld(
            action_args[0], action_args[1:]
        )

        return alfworld_action


if __name__ == "__main__":
    import alfworld
    import alfworld.agents.environment

    os.makedirs(LLMDPConfig.output_dir, exist_ok=True)

    with open(f"{LLMDPConfig.alfworld_config_path}/base_config.yaml") as reader:
        config = yaml.safe_load(reader)

    split = "eval_out_of_distribution"
    # split = "train"

    # UPDATE PATH TO ALFWORLD DATA
    for k in config:
        for i, j in config[k].items():
            if type(j) == str and j.startswith("$"):
                config[k][i] = config[k][i].replace(
                    "$ALFWORLD_DATA", LLMDPConfig.alfworld_data_path
                )

    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval=split
    )
    env = env.init_env(batch_size=1)

    def process_ob(ob):
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2 :]
        return ob

    NUM_GAMEFILES = len(env.gamefiles)

    logger = get_logger(
        f"{LLMDPConfig.output_dir}/{LLMDPConfig.name}_{LLMDPConfig.llm_model}.log"
    )

    def llmdp_run(agent) -> tuple[int, int]:
        last_observation = ""
        for i in range(1, LLMDPConfig.max_steps):
            try:
                action = agent.take_action(last_observation=last_observation)
            except Exception as e:
                logger.error(f"Error in taking action {str(e)}")
                logger.info(str(agent.scene_objects))
                logger.info(str(agent.plan))
                return 0, i

            logger.info(f"{i} Action: {action}")
            observation, reward, done, info = env.step([action])
            observation, reward, done = (
                process_ob(observation[0]),
                info["won"][0],
                done[0],
            )
            last_observation = observation
            logger.info(f"{i} Obs: {last_observation}")
            if done:
                return reward, i
        return 0, i

    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }
    cnts = [0] * 6
    rs = [0] * 6
    results = []

    for n in range(NUM_GAMEFILES):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        scene_observation, task_description = ob.split("\n")
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        logger.info(name)
        logger.info(scene_observation)
        logger.info(task_description)

        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                try:
                    agent = LLMDPAgent(
                        scene_observation,
                        task_description,
                        logger=logger,
                        use_llm_search=LLMDPConfig.use_llm_search,
                    )
                    r, length = llmdp_run(agent)
                except Exception as e:
                    logger.error(f"{str(e)}")
                    r = 0
                    length = 0

                rs[i] += r
                cnts[i] += 1

                results.append(
                    {
                        "task": v,
                        "success": r,
                        "length": length,
                        "llm_tokens": agent.llm_tokens_sent,
                        "llm_chat_tokens": agent.llm_chat_tokens_sent,
                    }
                )
                logger.info(f"# Tokens used: {agent.llm_tokens_sent}")
                out_log = f"# {n + 1} r: {r} rs: {rs} cnts: {cnts} sum(rs) / sum(cnts): {sum(rs) / sum(cnts)}"
                logger.info(out_log)
                break

        logger.info("------------\n")
        # save results
        with open(
            f"{LLMDPConfig.output_dir}/{LLMDPConfig.name}_{LLMDPConfig.llm_model}_results.json",
            "w",
        ) as f:
            json.dump(results, f)
