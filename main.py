import json
import random
import os
from logging import Logger

import yaml

from config import LLMDPConfig
from logger import get_logger
from llm_utils import llm_cache
from llmdp import LLMDPAgent


def llmdp_run(agent, logger: Logger) -> tuple[int, int]:
    """
    Using the LLM-DP agent to navigate the environment
    """
    last_observation = ""
    for i in range(1, 50):
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
            return reward, i, agent.llm_tokens_used
    return 0, i, agent.llm_tokens_used


def alfworld_react_run_chat(
    scene_observation: str, task_description: str, task_type: str, logger: Logger
) -> tuple[int, int, int]:
    """
    React gpt-3.5-turbo (chatGPT) version of the alfworld_react_run function
    """

    # load few shot prompts
    folder = "./prompts/"
    prompt_file = "alfworld_3prompts_react_chat.json"
    with open(folder + prompt_file, "r") as f:
        d = json.load(f)

    system_prompt = [
        {
            "role": "system",
            "content": "Interact with a household to solve a task."
            + " Only reply with > followed by the action to take or 'think'."
            + " Do not apologize."  # necessary for ChatGPT (more apologetic than GPT-3)
            + " Follow the format of the two examples below.",
        }
    ]
    description_prompt = [
        {"role": "user", "content": f"DESCRIPTION {scene_observation}"}
    ]
    task_prompt = [{"role": "user", "content": f"TASK: {task_description}"}]

    chat_prompts = (
        system_prompt
        + d[f"react_{task_type}_1"]
        + d[f"react_{task_type}_0"]
        + description_prompt
        + task_prompt
    )

    llm_tokens_used = 0
    for i in range(1, 50):
        action, token_usage = llm_cache(chat_prompts)
        llm_tokens_used += token_usage["total_tokens"]
        observation, reward, done, info = env.step([action.replace(">", "").strip()])
        observation, reward, done = (
            process_ob(observation[0]),
            info["won"][0],
            done[0],
        )
        # check if the agent is apologizing (will get stuck in a loop)
        if "I apologize" in action:
            return 0, i, llm_tokens_used

        if action.startswith("> think:"):
            observation = "OK."

        # change 'in' or 'on' to 'in/on'
        # this is a quirk of Alfworld which ChatGPT does not handle well
        # since it just picks the most appropriate word (in or on)
        # depending on the context
        if "put " in action and (" in " in action or " on " in action):
            action = action.replace(" in ", " in/on ")

        logger.info(f"{i} Action: {action}")
        logger.info(f"{i} Obs: {observation}")
        chat_prompts.append({"role": "assistant", "content": action})
        chat_prompts.append({"role": "user", "content": observation})
        if done:
            return reward, i, llm_tokens_used

    return 0, i, llm_tokens_used


if __name__ == "__main__":
    import alfworld
    import alfworld.agents.environment

    os.makedirs(LLMDPConfig.output_dir, exist_ok=True)

    if LLMDPConfig.use_react_chat:
        run_name = f"{LLMDPConfig.output_dir}/react_chat"
    else:
        run_name = f"{LLMDPConfig.output_dir}/{LLMDPConfig.name}_{LLMDPConfig.top_n}"

    with open(f"{LLMDPConfig.alfworld_config_path}/base_config.yaml") as reader:
        config = yaml.safe_load(reader)

    split = "eval_out_of_distribution"

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

    logger = get_logger(f"{run_name}.log")

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

    # load previous results (if llm errors - useful for resuming)
    prev_results = []
    # with open(f"{run_name}.json", "rb") as f:
    # prev_results = json.loads(f.read())

    for n in range(NUM_GAMEFILES):
        # Set seed for reproducibility
        random.seed(LLMDPConfig.seed)

        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        scene_observation, task_description = ob.split("\n")
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        logger.info(name)
        logger.info(scene_observation)
        logger.info(task_description)

        # only run if prev failed (or if not in prev)
        if n < len(prev_results) and prev_results[n]["success"]:
            results.append(prev_results[n])
            continue

        for i, (k, task_type) in enumerate(prefixes.items()):
            if name.startswith(k):
                try:
                    if LLMDPConfig.use_react_chat:
                        r, length, llm_tokens_used = alfworld_react_run_chat(
                            scene_observation, task_description, task_type, logger
                        )
                    else:
                        agent = LLMDPAgent(
                            scene_observation,
                            task_description,
                            logger=logger,
                            use_llm_search=LLMDPConfig.use_llm_search,
                            top_n=LLMDPConfig.top_n,
                        )
                        r, length, llm_tokens_used = llmdp_run(agent)
                except Exception as e:
                    logger.error(f"{str(e)}")
                    r = 0
                    length = 0
                    llm_tokens_used = 0

                rs[i] += r
                cnts[i] += 1

                results.append(
                    {
                        "task": task_type,
                        "success": r,
                        "length": length,
                        "llm_tokens": llm_tokens_used,
                    }
                )
                logger.info(f"# Tokens used: {llm_tokens_used}")
                out_log = f"# {n + 1} r: {r} rs: {rs} cnts: {cnts} sum(rs) / sum(cnts): {sum(rs) / sum(cnts)}"
                logger.info(out_log)
                break

        logger.info("------------\n")
        # save results
        with open(f"{run_name}.json", "w") as f:
            json.dump(results, f)
