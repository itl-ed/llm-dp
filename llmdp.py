import json
import os
import pickle
import random
import re
import time
import logging
from collections import defaultdict

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


FIND_OBJECTS_PROMPT = """Given a task outline what object the planner needs to find/look for.
Your task is to: cool some egg and put it in garbagecan.
egg
Your task is to: put a hot plate in fridge.
plate
Your task is to: clean some soapbar and put it in countertop.
soapbar
Your task is to: put a soapbottle in cabinet.
soapbottle
Your task is to: examine the cd with the desklamp.
cd, desklamp
Your task is to: put some toiletpaper on toiletpaperhanger.
toiletpaper
Your task is to: put two cellphone in bed.
cellphone, cellphone"""

FIND_RECEPTACLES_PROMPT = """Given a task outline what receptacles the planner needs to find/look (if any). If none, reply with 'none'.
Your task is to: cool some egg and put it in garbagecan.
garbagecan
Your task is to: put a hot plate in fridge.
fridge
Your task is to: clean some soapbar and put it in countertop.
countertop
Your task is to: put a soapbottle in cabinet.
cabinet
Your task is to: examine the cd with the desklamp.
none
Your task is to: put some toiletpaper on toiletpaperhanger.
toiletpaperhanger
Your task is to: put two cellphone in bed.
bed
"""

GENERATE_GOAL_PROMPT = """(define (domain alfred)
(:requirements :typing)
(:types
receptacle object rtype
)
(:constants
SinkBasinType - rtype
MicrowaveType - rtype
FridgeType - rtype
)
(:predicates
(atReceptacleLocation ?r - receptacle) ; true if the robot is at the receptacle location
(inReceptacle ?o - object ?r - receptacle) ; true if object ?o is in receptacle ?r
(openable ?r - receptacle) ; true if a receptacle is openable
(opened ?r - receptacle) ; true if a receptacle is opened
(isLight ?o - object) ; true if an object is light source
(examined ?o - object ?l - object) ; whether the object has been looked at with light
(holds ?o - object) ; object ?o is held by robot
(isClean ?o - object) ; true if the object has been cleaned in sink
(isHot ?o - object) ; true if the object has been heated up
(isCool ?o - object) ; true if the object has been cooled
(receptacleType ?r - receptacle ?t - rtype) ; true if the receptacle is of type ?t
)
)
TASK: Your task is to: put a clean plate in microwave.
(:objects
microwave-1 receptacle
plate-1 - object
)
(:goal
(exists (?t - object)
(and (inReceptacle ?t microwave-1)
(isClean ?t)
)))
TASK: Your task is to: examine an alarmclock with the desklamp
(:objects
alarmclock-1 - object
desklamp-1 - object
)
(:goal
(exists (?t - object ?l - object) 
(examined ?t ?l)
))
TASK: Your task is to: put two cellphone in bed
(:objects
bed-1 receptacle
cellphone-1 - object
cellphone-2 - object
)
(:goal
(exists (?t1 - object ?t2 - object)
(and (inReceptacle ?t1 bed-1)
(inReceptacle ?t2 bed-1)
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
    ) -> None:
        self.initial_scene_observation = initial_scene_observation
        self.task_description = task_description
        self.env_id = env_id
        self.llm_tokens_sent = 0
        self.llm_chat_tokens_sent = 0
        self.logger = logger
        self.use_llm_search = use_llm_search

        # identify relevant objects and receptacle from task_description for goal
        self.relevant_task_objects = self.get_relevant_task_objects(task_description)
        # set of object types to look out for (e.g. apple, bread, etc.)
        self.revevant_task_object_type = set(
            [o.split("-")[0] for o in self.relevant_task_objects]
        )
        # identify relevant receptacles from task_description for goal
        self.relevant_task_receptacles = self.get_relevant_task_receptacles(
            task_description
        )
        # get PDDL objects from scene_observation
        self.scene_receptacles = self.find_receptacles_from_scene_observation(
            initial_scene_observation
        )

        # populate scene_objects with all objects/receptacles in scene
        self.scene_objects = defaultdict(dict)
        for r in self.scene_receptacles:
            self.scene_objects[r] = self.get_receptacle_attributes(r)
            self.scene_objects[r]["type"] = "receptacle"
            self.scene_objects[r]["seen"] = False
            if r in self.relevant_task_receptacles:
                self.scene_objects[r]["relevant"] = True

        for o in self.relevant_task_objects:
            self.scene_objects[o]["type"] = "object"
            self.scene_objects[o]["relevant"] = True
            self.scene_objects[o]["seen"] = False
            if "lamp" in o:
                self.scene_objects[o]["isLight"] = True

        self.pddl_goal = self.get_pddl_goal()
        self.logger.info(f"PDDL GOAL: {self.pddl_goal}")

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

    def get_relevant_task_objects(self, task_description: str) -> list[str]:
        """
        Given a task description, return the object(s) that the planner needs to look for.
        """
        prompt = f"{FIND_OBJECTS_PROMPT}\n{task_description}\n"
        task_objects_str = self.llm(prompt, stop=["\n"]).strip()
        self.logger.info(task_objects_str)
        task_objects = []
        if "," in task_objects_str:
            task_objects = task_objects_str.split(", ")
        else:
            task_objects = [task_objects_str]
        # add -1, -2, -3, etc. to the end of each object
        return self.get_object_count_id(task_objects)

    def get_relevant_task_receptacles(self, task_description: str) -> list[str]:
        """
        Given a task description, return the receptacle that the planner needs to look at.
        """
        prompt = f"{FIND_RECEPTACLES_PROMPT}\n{task_description}\n"
        task_receptacles_str = self.llm(prompt, stop=["\n"]).strip()
        self.logger.info(task_receptacles_str)
        if task_receptacles_str.strip() == "none":
            return []

        task_receptacles = []
        if "," in task_receptacles_str:
            task_receptacles = task_receptacles_str.split(", ")
        else:
            task_receptacles = [task_receptacles_str]
        # add -1, -2, -3, etc. to the end of each receptacle
        return self.get_object_count_id(task_receptacles)

    def get_task_goal(
        self,
        task_description: str,
        relevant_objects: list[str],
        relevant_receptacles: list[str],
    ) -> list[str]:
        """
        Given a task description, relevant objects/receptacles return the PDDL goal.
        """

        objects_str = ""
        for relevant_receptacle in relevant_receptacles:
            objects_str += f"{relevant_receptacle} - receptacle\n"
        for relevant_object in relevant_objects:
            objects_str += f"{relevant_object} - object\n"

        prompt = f"{GENERATE_GOAL_PROMPT}\nTASK: {task_description}\n(:objects\n{objects_str}\n)\n(goal:"
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
        attributes = {}

        # predicate special types
        if "sink" in receptacle:
            receptacleType = "SinkBasinType"
            attributes["receptacleType"] = receptacleType
        elif "microwave" in receptacle:
            receptacleType = "MicrowaveType"
            attributes["receptacleType"] = receptacleType
        elif "fridge" in receptacle:
            receptacleType = "FridgeType"
            attributes["receptacleType"] = receptacleType

        # predicate openable
        if (
            "microwave" in receptacle
            or "fridge" in receptacle
            or "drawer" in receptacle
            or "cabinet" in receptacle
            or "safe" in receptacle
        ):
            attributes["openable"] = True

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
        completed_pddl_goal = self.get_task_goal(
            self.task_description,
            self.relevant_task_objects,
            self.relevant_task_receptacles,
        )
        return f"(:goal {completed_pddl_goal}"

    def get_pddl_objects(self) -> str:
        # get all objects/receptacles in scene
        objects_str = "".join(
            [f"{o} - {atts['type']}\n" for o, atts in self.scene_objects.items()]
        )
        return f"(:objects {objects_str})\n"

    def get_pddl_init(self) -> str:
        # fill in known predicates from observation
        predicates = ""
        for r, atts in self.scene_objects.items():
            for att, val in atts.items():
                if att in ["type", "relevant", "seen"] or val is False:
                    continue
                if val is True:
                    predicates += f"({att} {r})\n"
                else:
                    predicates += f"({att} {r} {val})\n"

        unseen_receptacles = [
            r
            for r, atts in self.scene_objects.items()
            if not atts["seen"] and atts["type"] == "receptacle"
        ]
        unseen_objects = [
            o
            for o, atts in self.scene_objects.items()
            if not atts["seen"] and atts["type"] == "object"
        ]

        # if unseen objects, use LLM to guess receptacle
        if len(unseen_objects) > 0:
            # We can assume that the task is not to put an object in place that it already is
            if len(unseen_receptacles) > 1:
                unseen_receptacles = list(
                    set(unseen_receptacles) - set(self.relevant_task_receptacles)
                )

            if len(unseen_receptacles) > 1:
                # Use LLM to guess which receptacle
                if self.use_llm_search:
                    likely_receptacle, llm_tokens = llm_find_objects(
                        unseen_objects, unseen_receptacles, self.logger
                    )
                    self.llm_chat_tokens_sent += llm_tokens
                # Use a random unseen receptacle
                else:
                    likely_receptacle = random.choice(unseen_receptacles)

            else:
                # If there is only one unseen receptacle, use that
                likely_receptacle = unseen_receptacles[0]

            for o in unseen_objects:
                predicates += f"(inReceptacle {o} {likely_receptacle})\n"

        pddl_init = f"(:init {predicates})\n"
        return pddl_init

    def get_pddl_problem(
        self,
    ) -> str:
        # construct to PDDL problem.pddl
        problem = (
            "(define (problem alf)\n(:domain alfred)\n"
            + f"{self.get_pddl_objects()}{self.get_pddl_init()}{self.pddl_goal})"
        )
        return problem

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
        # NOTE: this is using the symbolic :effect of the action
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
            # NOTE: we ignore seen objects that are not relevant to the task
            # We also add objects of the same type as the target object
            for obj in objects:
                if obj in self.scene_objects:
                    self.scene_objects[obj]["seen"] = True
                    self.scene_objects[obj]["inReceptacle"] = receptacle
                elif obj.split("-")[0] in self.revevant_task_object_type:
                    self.scene_objects[obj] = {
                        "type": "object",
                        "relevant": True,
                        "seen": True,
                        "inReceptacle": receptacle,
                    }

        return scene_changed

    def get_plan(self) -> list[str]:
        problem = self.get_pddl_problem()
        return call_lakpt_solver(problem, solver="bfs_f", logger=self.logger)

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


# Alternate way of processing observations using LLM (not used because slower/costly)
# def process_obs(obs: str) -> dict:
#     prompt = """Convert each observation into a json dict, such that it can be parsed with the python json.loads function. If a receptacle is closed, return an empty dictionary.
# On the sinkbasin 1, you see a dishsponge 2, a spatula 3, and a tomato 1.
# {"sinkbasin-1": ["dishsponge-2","spatula-3", "tomato-1"]}
# You open the cabinet 4. The cabinet 4 is open. In it, you see nothing.
# {"cabinet-4": []}
# You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, a egg 3, a egg 2, and a pan 2.
# {"fridge-1": [ "apple-2", "egg-3", "egg-2", "pan-2"]}
# On the garbagecan 1, you see a bread 3.
# {"garbagecan-1": ["bread-3"]}
# The microwave 1 is closed.
# {}
# On the countertop 1, you see a apple 1, a bread 2, a butterknife 1, a mug 2, a potato 2, and a soapbottle 2.
# {"countertop-1": ["apple-1", "bread-2", "butterknife-1", "mug-2", "potato-2", "soapbottle-2"]}
# On the coffeemachine 1, you see a mug 1.
# {"coffeemachine-1": ["mug-1"]}
# The drawer 2 is closed.
# {}
# """
#     prompt += f"{obs}\n"
#     out = llm(prompt, stop=["\n"])
#     try:
#         out = json.loads(out)
#     except json.decoder.JSONDecodeError:
#         print(f"Could not convert observation {obs} into dict from {out}.")
#         out = {}
#     return out
