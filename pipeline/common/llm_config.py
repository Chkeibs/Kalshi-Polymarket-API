import os
import time
import requests
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- CONFIGURATION ---
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17")
API_KEY = os.environ.get("OPENAI_API_KEY")
PROMPT_LOG = os.path.join(ROOT_DIR, "logs", "llm_prompts.log")
BASE_DELAY = 1.0  # Seconds between calls

def log_llm_prompt(prompt, context):
    """Log the LLM prompt to a text file for audit."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"--- [{timestamp}] CONTEXT: {context} ---\n{prompt}\n\n"
    try:
        os.makedirs(os.path.dirname(PROMPT_LOG), exist_ok=True)
        with open(PROMPT_LOG, "a") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error logging prompt: {e}")

def safe_call_llm(prompt, system_message="You are an expert market data analyst.", response_format=None, context="General", model_name=None):
    """
    Call the LLM API with exponential backoff and base delay.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    log_llm_prompt(prompt, context)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    selected_model = model_name or MODEL_NAME
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }
    if selected_model.startswith("gpt-5.4"):
        payload["reasoning_effort"] = "none"
    elif not selected_model.startswith("gpt-5"):
        payload["temperature"] = 0.0
    
    if response_format:
        payload["response_format"] = response_format

    max_retries = 5
    backoff_time = 2.0
    
    for attempt in range(max_retries):
        try:
            # Respect base delay
            time.sleep(BASE_DELAY)
            
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            
            if resp.status_code == 429:
                print(f"[Rate Limit] Attempt {attempt+1} failed. Backing off for {backoff_time}s...")
                time.sleep(backoff_time)
                backoff_time *= 2
                continue
                
            print(f"[LLM Error] Status: {resp.status_code} | Response: {resp.text}")
            return None
            
        except Exception as e:
            print(f"[LLM Exception] {e}")
            time.sleep(backoff_time)
            backoff_time *= 2
            
    return None
