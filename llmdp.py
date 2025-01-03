import random
import re
from typing import Literal

from collections import Counter, defaultdict

from utils.planner import parallel_lapkt_solver
from utils.llm_utils import llm_cache

GENERATE_GOAL_PROMPT = [
    {
        "role": "system",
        "content": """(define (domain alfred)
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
))""",
    },
    {
        "role": "user",
        "content": "Your task is to: put a clean plate in microwave.",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - plate ?r - microwave)
(and (inReceptacle ?t ?r)
(isClean ?t)
)))""",
    },
    {
        "role": "user",
        "content": "Your task is to: examine an alarmclock with the desklamp",
    },
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t - alarmclock ?l - desklamp)
(and (examined ?t ?l) (holds ?t)
)))""",
    },
    {"role": "user", "content": "Your task is to: put two cellphone in bed"},
    {
        "role": "assistant",
        "content": """(:goal
(exists (?t1 - cellphone ?t2 - cellphone ?r - bed)
(and (inReceptacle ?t1 ?r)
(inReceptacle ?t2 ?r)
(not (= ?t1 ?t2))
)))""",
    },
]


class LLMDPAgent:
    """
    Alfworld agent that uses the LLMDP planner to generate a plan to complete a task.
    """

    def __init__(
        self,
        initial_scene_observation: str,
        task_description: str,
        logger=None,
        sample: Literal["llm", "random"] = "llm",
        top_n=3,
        random_fallback=False,
        temperature=0.0,
    ) -> None:
        self.initial_scene_observation = initial_scene_observation
        self.task_description = task_description
        self.llm_tokens_used = 0
        self.logger = logger
        self.sample = sample
        self.top_n = top_n
        self.random_fallback = random_fallback
        self.temperature = temperature

        # get PDDL objects from scene_observation
        scene_receptacles = self.find_receptacles_from_scene_observation(
            initial_scene_observation
        )

        # populate scene_objects with all objects/receptacles in scene
        self.scene_objects = defaultdict(dict)
        for r in scene_receptacles:
            self.scene_objects[r] = self.get_receptacle_attributes(r)

        # use LLM to translate task description to PDDL goal
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

                # initialise beliefs about the object's location
                self.scene_objects[name]["beliefs"] = {}

                # if not a receptacle, then it's location is unknown
                if "isReceptacle" not in self.scene_objects[name]:
                    # all receptacles are possible locations for an object
                    self.scene_objects[name]["beliefs"]["inReceptacle"] = (
                        scene_receptacles.copy()
                    )

                if "lamp" in name:
                    self.scene_objects[name]["isLight"] = True

        self.actions_taken = []

    @staticmethod
    def process_obs(observation: str) -> dict:
        """
        No LLM version of process_obs prefered for efficiency.
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
        """
        Given a task description return the PDDL goal using an LLM.
        """
        prompt_messages = GENERATE_GOAL_PROMPT + [
            {
                "role": "user",
                "content": self.task_description,
            }
        ]
        pddl_goal, token_usage = llm_cache(
            prompt_messages, stop=None, temperature=self.temperature
        )
        self.llm_tokens_used += token_usage["total_tokens"]
        return pddl_goal

    def get_pddl_belief_predicate(
        self, init_str: str, belief_predicate: str, belief_values: list[str], top_n: 1
    ) -> list[str]:
        """
        Uses the LLM to predict the most likely values for an unknown predicate in the environment.

        Given a belief predicate (e.g., 'inReceptacle plate ?') and a list of possible values (e.g., ['fridge', 'countertop']),
        this function queries the LLM to select the top N most likely values for the unknown variable (?).
    
        The LLM leverages its semantic knowledge of the world to infer plausible values based on the observed environment
        and the context provided by the initial state (init_str).
    
        Args:
            init_str (str): A string representation of the current observed environment state.
            belief_predicate (str): The predicate to predict (e.g., 'inReceptacle plate ?').
            belief_values (list[str]): A list of possible values for the unknown variable (?).
            top_n (int): The number of most likely values to return.
    
        Returns:
            list[str]: A list of the top N most likely values for the unknown variable (?).
    
        Example:
            >>> init_str = "The fridge is closed. The countertop is empty."
            >>> belief_predicate = "inReceptacle plate ?"
            >>> belief_values = ["fridge", "countertop", "microwave"]
            >>> top_n = 2
            >>> get_pddl_belief_predicate(init_str, belief_predicate, belief_values, top_n)
            ['fridge', 'countertop']
        """
        user_prompt = (
            f"Predict: {belief_predicate}\n"
            + f"Select the top {top_n} likely items for ? from the list:"
            + f"{sorted(belief_values)}\n"
            + "Return a parsable python list of choices."
        )
        prompt_messages = [
            {"role": "system", "content": f"Observed Environment\n{init_str}"},
            {"role": "user", "content": user_prompt},
        ]
        selected_values, token_usage = llm_cache(
            prompt_messages, stop=None, temperature=self.temperature
        )
        self.llm_tokens_used += token_usage["total_tokens"]
        # parse the selected values as list
        try:
            selected_values = re.findall(r"'(.*?)'", selected_values)
        except Exception as e:
            self.logger.info(f"Error parsing selected values: {selected_values}")
            raise e
        return selected_values

    def get_pddl_objects(self) -> str:
        # get all objects/receptacles in scene
        objects_str = "".join(
            [f"{o} - {atts['type']}\n" for o, atts in self.scene_objects.items()]
        )
        return f"(:objects {objects_str})\n"

    def get_pddl_init(self, sample="random") -> list[str]:
        # fill in known predicates from observation
        known_predicates = ""

        # known predicates
        for r, atts in self.scene_objects.items():
            for att, val in atts.items():
                if att in ["type", "beliefs"] or val is False:
                    continue
                if val is True:
                    known_predicates += f"({att} {r})\n"
                else:
                    known_predicates += f"({att} {r} {val})\n"

        # dynamic predicates (World Beliefs)
        belief_predicates = [known_predicates] * self.top_n
        for o, atts in self.scene_objects.items():
            if "beliefs" in atts:
                for belief_attribute in atts["beliefs"]:
                    options = atts["beliefs"][belief_attribute]

                    # sample N different worlds for each belief
                    if sample == "random":
                        sampled_beliefs = random.choices(options, k=self.top_n)
                    elif sample == "llm":
                        # Use LLM to guess which receptacle
                        sampled_beliefs = self.get_pddl_belief_predicate(
                            init_str=known_predicates,
                            belief_predicate=f"({belief_attribute} {o} ?)",
                            belief_values=options,
                            top_n=self.top_n,
                        )
                        # ensure that the sampled belief is in the list of options
                        hallucination_set = set(sampled_beliefs) - set(options)
                        for i, element in enumerate(sampled_beliefs):
                            if element in hallucination_set:
                                self.logger.warning(
                                    f"Hallucination: Sampled belief {element} not in {options}"
                                )
                                sampled_beliefs[i] = random.choice(options)
                    else:
                        raise ValueError(f"Unknown sample method: {sample}")

                    # append the sampled belief predicate to each world state
                    belief_predicates = list(
                        map(
                            lambda x, s: x + f"({belief_attribute} {o} {s})\n",
                            belief_predicates,
                            sampled_beliefs,
                        )
                    )

        return list(
            set([f"(:init {predicates})\n" for predicates in belief_predicates])
        )

    def get_pddl_problem(self, sample: Literal["llm", "random"] = "llm") -> list[str]:
        # get n different init configurations
        inits = self.get_pddl_init(sample=sample)

        problems = []
        for init in inits:
            # construct to PDDL problem.pddl
            problems.append(
                "(define (problem alf)\n(:domain alfred)\n"
                + f"{self.get_pddl_objects()}{init}{self.pddl_goal})"
            )
        return problems

    def update_observation(self, observation: str) -> bool:
        # case for initial observation
        if observation == "":
            return True

        scene_obs = {}
        scene_changed = False

        # use last action to update scene_objects
        action_args = self.actions_taken[-1]
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
        for receptacle, seen_objects in scene_obs.items():
            # if you can see objects in receptacle, it must be opened
            if "openable" in self.scene_objects[receptacle]:
                self.scene_objects[receptacle]["opened"] = True

            # update beliefs
            # all objects not observed at this receptacle cannot be believed to be in it
            for obj in self.scene_objects.keys():
                if (
                    obj not in seen_objects
                    and "beliefs" in self.scene_objects[obj]
                    and "inReceptacle" in self.scene_objects[obj]["beliefs"]
                    and receptacle in self.scene_objects[obj]["beliefs"]["inReceptacle"]
                ):
                    self.scene_objects[obj]["beliefs"]["inReceptacle"].remove(
                        receptacle
                    )

            # update inReceptacle for all objects observed at this receptacle
            for obj in seen_objects:
                self.scene_objects[obj]["type"] = obj.split("-")[0]
                self.scene_objects[obj]["inReceptacle"] = receptacle
                if (
                    "beliefs" in self.scene_objects[obj]
                    and "inReceptacle" in self.scene_objects[obj]["beliefs"]
                ):
                    del self.scene_objects[obj]["beliefs"]["inReceptacle"]
                if "lamp" in obj:
                    self.scene_objects[obj]["isLight"] = True

        return scene_changed

    def get_plan(self) -> list[str]:
        # Plan Generator
        problems = self.get_pddl_problem(sample=self.sample)
        plans = parallel_lapkt_solver(problems, logger=self.logger)

        # In some cases the LLM fails to generate valid states
        # (e.g. if instantiates only goal satisfying states)
        if self.random_fallback and len(plans) == 0:
            self.logger.warning("No plans found: sampling randomly.")
            problems = self.get_pddl_problem(sample="random")
            plans = parallel_lapkt_solver(problems, logger=self.logger)

        # Action Selector: greedy selection strategy
        return min(plans, key=len)

    def take_action(self, last_observation="") -> str:
        # sometimes move action doesn't trigger observation (e.g. if already close)
        # this is a flaw with Alfworld, so we manually trigger an observation
        if (
            last_observation == "Nothing happens."
            and self.actions_taken[-1][0] == "gotoreceptacle"
        ):
            return f"examine {self.actions_taken[-1][1]}".replace("-", " ")
        # if last action was not move, then we should probably have observed something
        elif last_observation == "Nothing happens.":
            self.logger.warning(f"Invalid Action: {self.actions_taken[-1][0]} failed.")

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
        self.actions_taken.append(action_args)

        alfworld_action = self.convert_pddl_action_to_alfworld(
            action_args[0], action_args[1:]
        )

        return alfworld_action
