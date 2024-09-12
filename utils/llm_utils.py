import json
import os
import pickle
import time

from openai import OpenAI
from openai._exceptions import RateLimitError
from utils.config import LLMDPConfig


client = OpenAI(api_key=LLMDPConfig.openai_api_key)


def llm(llm_messages: list[str], stop=None) -> tuple[str, dict[str, int]]:
    try:
        completion = client.chat.completions.create(
            model=LLMDPConfig.llm_model,
            messages=llm_messages,
            temperature=0.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return completion.choices[0].message.content, dict(completion.usage)
    except RateLimitError:
        time.sleep(10)
        return llm(llm_messages, stop=stop)


def llm_cache(
    llm_messages: list[dict[str]],
    stop=None,
    temperature=0,
) -> tuple[str, dict[str, int]]:
    # If temperature is greater than 0, then skip cache
    if temperature > 0:
        return llm(llm_messages, stop=stop)

    # Cache the openai responses in pickle file
    cache_file = (
        f"{LLMDPConfig.output_dir}/llm_responses_{LLMDPConfig.llm_model}.pickle"
    )
    llm_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            llm_responses = pickle.load(f)

    key = json.dumps(llm_messages)
    if key not in llm_responses:
        print("Not in cache")
        generated_content, token_usage = llm(llm_messages, stop=stop)
        llm_responses[key] = (generated_content, token_usage)
        with open(cache_file, "wb") as f:
            pickle.dump(llm_responses, f)

    return llm_responses[key]
