"""
DataClean-Ops Inference Script
OpenEnv-compatible inference with LLM agent

Output format: [START], [STEP], [END] with structured logging.
Uses OpenAI Client with MODEL_NAME, API_BASE_URL, HF_TOKEN environment variables.

Based on sample inference.py format from Meta OpenEnv Challenge.
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Configuration
TASK_NAME = os.environ.get("TASK", "easy")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 2.0

BENCHMARK = "DataClean-Ops"


def log_start(task: str, env: str, model: str):
    """Log start of episode."""
    print(f"[START] Task={task}, Env={env}, Model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    """Log each step."""
    error_str = f", Error={error}" if error else ""
    print(f"[STEP] Step={step}, Action={action}, Reward={reward:+.4f}, Done={done}{error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    """Log end of episode."""
    print(f"[END] Success={success}, Steps={steps}, Score={score:.4f}, TotalReward={sum(rewards):.4f}", flush=True)


@dataclass
class EpisodeResult:
    episode: int
    task: str
    total_steps: int
    total_reward: float
    success: bool
    duration: float
    score: float
    actions: List[str]
    rewards: List[float]


class LLMAgent:
    """LLM-powered agent using OpenAI Client."""
    
    def __init__(self):
        self.model_name = MODEL_NAME
        self.api_base_url = API_BASE_URL
        self.api_key = HF_TOKEN
        
        self.client = None
        self.use_fallback = True
        self.history = []
        
        if self.api_key and len(self.api_key) > 10:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
                self.use_fallback = False
                print(f"[INFO] Using OpenAI API: {self.model_name}", flush=True)
            except ImportError:
                self.use_fallback = True
        else:
            print("[INFO] No API key - using fallback agent", flush=True)
    
    def select_action(self, observation, task: str, step: int) -> str:
        """Select best action based on current state."""
        tables = observation.get("tables", {})
        max_steps = observation.get("max_steps", 20)
        
        table_info = []
        for name, tbl in tables.items():
            table_info.append(f"{name}: {tbl.get('rows', 0)} rows, {tbl.get('null_pct', 0):.1f}% nulls, {tbl.get('dupes', 0)} dupes")
        
        state_desc = f"Task: {task}, Step: {step}/{max_steps}. Tables: {'; '.join(table_info)}"
        
        if not self.use_fallback:
            prompt = f"""
You are a data cleaning assistant. Given the current state:
{state_desc}

Available actions: clean_nulls, format_date, drop_duplicates, merge_tables, remove_outliers, normalize_currency, validate

Choose the SINGLE best action that will improve data quality. Return only the action name.
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.3
                )
                action = response.choices[0].message.content.strip().lower()
                
                valid_actions = ["clean_nulls", "format_date", "drop_duplicates", "merge_tables", 
                               "remove_outliers", "normalize_currency", "validate"]
                for valid in valid_actions:
                    if valid in action:
                        self.history.append(f"Step {step}: {action}")
                        return valid
                return "validate"
            except Exception as e:
                print(f"[WARN] API error: {e}", flush=True)
        
        # Fallback: rule-based agent
        return self._fallback_action(task, step)
    
    def _fallback_action(self, task: str, step: int) -> str:
        """Rule-based action selection."""
        if not hasattr(self, 'action_index'):
            self.action_index = 0
        
        task_actions = {
            "easy": ["clean_nulls", "clean_nulls", "validate"],
            "medium": ["format_date", "normalize_currency", "drop_duplicates", "validate"],
            "hard": ["remove_outliers", "merge_tables", "clean_nulls", "validate"]
        }
        
        actions = task_actions.get(task, ["clean_nulls", "validate"])
        
        if self.action_index < len(actions):
            action = actions[self.action_index]
            self.action_index += 1
            self.history.append(f"Step {step}: {action} (fallback)")
            return action
        
        self.history.append(f"Step {step}: validate (fallback)")
        return "validate"


async def run_async_episode(task: str, agent: LLMAgent, max_steps: int = 20) -> EpisodeResult:
    """Run one episode on the environment (async version)."""
    from env import create_env
    from models import Action, ActionType, ActionParams
    
    start_time = time.time()
    env = create_env(task)
    result = env.reset()
    
    observation = result.model_dump() if hasattr(result, 'model_dump') else {
        "task": result.task,
        "step": result.step,
        "max_steps": result.max_steps,
        "tables": {name: {"rows": t.rows, "null_pct": t.null_pct, "dupes": t.dupes} 
                  for name, t in result.tables.items()}
    }
    
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    
    total_reward = 0.0
    steps = 0
    actions_taken = []
    rewards = []
    last_result = None
    
    for step in range(1, max_steps + 1):
        if env.done:
            break
        
        action_name = agent.select_action(observation, task, step)
        
        try:
            action_type = ActionType(action_name)
        except:
            action_type = ActionType.VALIDATE
        
        action = Action(action_type=action_type, params=ActionParams())
        last_result = env.step(action)
        
        reward = last_result.reward.value if last_result.reward else 0.0
        done = last_result.done
        
        rewards.append(reward)
        total_reward += reward
        steps = step
        actions_taken.append(action_name)
        
        observation = last_result.observation.model_dump() if hasattr(last_result.observation, 'model_dump') else {
            "task": last_result.observation.task,
            "step": last_result.observation.step,
            "max_steps": last_result.observation.max_steps,
            "tables": {name: {"rows": t.rows, "null_pct": t.null_pct, "dupes": t.dupes} 
                      for name, t in last_result.observation.tables.items()}
        }
        
        log_step(step=step, action=action_name, reward=reward, done=done, error=None)
        
        if done:
            break
    
    duration = time.time() - start_time
    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    
    try:
        if hasattr(env, 'close'):
            env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)
    
    return EpisodeResult(
        episode=1,
        task=task,
        total_steps=steps,
        total_reward=total_reward,
        success=success,
        duration=duration,
        score=score,
        actions=actions_taken,
        rewards=rewards
    )


def main():
    """Main inference entry point."""
    print(f"Starting DataClean-Ops Inference", flush=True)
    print(f"Task: {TASK_NAME}, Max Steps: {MAX_STEPS}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API Base: {API_BASE_URL}", flush=True)
    print(f"HF Token: {'SET' if HF_TOKEN else 'NOT SET'}", flush=True)
    
    start_time = time.time()
    
    agent = LLMAgent()
    result = asyncio.run(run_async_episode(TASK_NAME, agent, MAX_STEPS))
    
    elapsed = time.time() - start_time
    
    print("=" * 60, flush=True)
    print(f"Inference Complete", flush=True)
    print(f"Task: {result.task}", flush=True)
    print(f"Score: {result.score:.4f}", flush=True)
    print(f"Success: {result.success}", flush=True)
    print(f"Time: {elapsed:.2f}s", flush=True)
    print("=" * 60, flush=True)
    
    output = {
        "task": result.task,
        "episode": result.episode,
        "steps": result.total_steps,
        "reward": result.total_reward,
        "score": result.score,
        "success": result.success,
        "duration": elapsed,
        "actions": result.actions,
        "rewards": result.rewards
    }
    
    print(f"\n[RESULT] {json.dumps(output)}", flush=True)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())