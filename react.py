import json
import os
import random

import yaml

from config import LLMDPConfig
from logger import get_logger
from llm_utils import llm_cache

if __name__ == "__main__":
    import alfworld
    import alfworld.agents.environment

    folder = "./prompts/"
    prompt_file = "alfworld_3prompts_react_chat.json"
    with open(folder + prompt_file, "r") as f:
        d = json.load(f)

    os.makedirs(LLMDPConfig.output_dir, exist_ok=True)
    run_name = f"{LLMDPConfig.output_dir}/react_chat"

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

    def alfworld_react_run_chat(chat_prompts, to_print=True) -> tuple[int, int, int]:
        llm_tokens_used = 0
        for i in range(1, 50):
            action, token_usage = llm_cache(chat_prompts)
            llm_tokens_used += token_usage["total_tokens"]
            observation, reward, done, info = env.step(
                [action.replace(">", "").strip()]
            )
            observation, reward, done = (
                process_ob(observation[0]),
                info["won"][0],
                done[0],
            )
            if action.startswith("> think:"):
                observation = "OK."
            if to_print:
                logger.info(f"{i} Action: {action}")
                logger.info(f"{i} Obs: {observation}")
            chat_prompts.append({"role": "assistant", "content": action})
            chat_prompts.append({"role": "user", "content": observation})
            if done:
                return reward, i, llm_tokens_used
        return 0, i, llm_tokens_used

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

        for i, (k, v) in enumerate(prefixes.items()):
            llm_tokens_used = 0
            if name.startswith(k):
                try:
                    system_prompt = [
                        {
                            "role": "system",
                            "content": "Interact with a household to solve a task.  Only reply with > followed by the action to take or 'think'. DO NOT apologize. Here are two examples.",
                        }
                    ]
                    description_prompt = [
                        {"role": "user", "content": f"DESCRIPTION {task_description}"}
                    ]
                    task_prompt = [
                        {"role": "user", "content": f"TASK: {task_description}"}
                    ]
                    chat_prompts = (
                        system_prompt
                        + d[f"react_{v}_1"]
                        + d[f"react_{v}_0"]
                        + description_prompt
                        + task_prompt
                    )
                    r, length, llm_tokens_used = alfworld_react_run_chat(chat_prompts)
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
