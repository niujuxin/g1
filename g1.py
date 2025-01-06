import openai
import time
import os
import json
import dotenv
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


dotenv.load_dotenv()


@lru_cache()
def get_openai_client(model: str = "gpt-4o-mini") -> ChatOpenAI:

    # Check existance of `OPENAI_API_KEY` and `OPENAI_API_ENDPOINT` in environment.
    assert 'OPENAI_API_KEY' in os.environ, 'OPENAI_API_KEY is not set'
    assert 'OPENAI_API_ENDPOINT' in os.environ, 'OPENAI_API_ENDPOINT is not set'

    client = ChatOpenAI(
        model=model,
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_API_ENDPOINT'],
    )
    return client


def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    client = get_openai_client()
    try:
        if is_final_answer:
            response: BaseMessage = client.invoke(
                input=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.content
        else:
            response: BaseMessage = client.invoke(
                input=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.content)
        
    except Exception as e:
        if is_final_answer:
            return {"title": "Error", "content": f"Failed to generate final answer. Error: {str(e)}"}
        else:
            return {"title": "Error", "content": f"Failed to generate step. Error: {str(e)}", "next_action": "final_answer"}


SYSTEM_PROMPT = (
    "You are an expert AI assistant that explains your reasoning step by step. "
    "For each step, provide a title that describes what you're doing in that step, along with the content. "
    "Decide if you need another step or if you're ready to give the final answer. "
    "Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. "
    "USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. "
    "IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. "
    "CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. "
    "FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. "
    "DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."
    "\n\n"
    "Example of a valid JSON response:\n"
    "```json\n"
    "{\n"
    "    \"title\": \"Identifying Key Information\",\n"
    "    \"content\": \"To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...\",\n"
    "    \"next_action\": \"continue\"\n"
    "}\n"
    "```"
)


def generate_response(prompt, custom_client=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 4096, custom_client=custom_client)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
            break
        
        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({
        "role": "user", 
        "content": (
            "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. "
            "Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, "
            "such as exact formatting for free response or multiple choice."
        )
    })
    
    start_time = time.time()
    final_data = make_api_call(messages, 4096, is_final_answer=True, custom_client=custom_client)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))

    yield steps, total_thinking_time
