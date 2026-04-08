import os
import numpy as np
from typing import Optional, Dict, Any

from env import create_env, DataCleanOpsEnv
from models import Action, ActionType, ActionParams


class RLAgent:
    def __init__(self):
        pass
    
    def select_action(self, obs):
        raise NotImplementedError()


class HeuristicAgent(RLAgent):
    def __init__(self):
        super().__init__()
        self.step = 0
    
    def select_action(self, obs) -> Action:
        if obs.task == "easy":
            actions = [
                Action(action_type=ActionType.CLEAN_NULLS, params=ActionParams(strategy="drop")),
                Action(action_type=ActionType.CLEAN_NULLS, params=ActionParams(strategy="fill")),
                Action(action_type=ActionType.VALIDATE, params=ActionParams())
            ]
        elif obs.task == "medium":
            actions = [
                Action(action_type=ActionType.FORMAT_DATE, params=ActionParams()),
                Action(action_type=ActionType.NORMALIZE_CURRENCY, params=ActionParams()),
                Action(action_type=ActionType.DROP_DUPLICATES, params=ActionParams()),
                Action(action_type=ActionType.VALIDATE, params=ActionParams())
            ]
        else:
            actions = [
                Action(action_type=ActionType.REMOVE_OUTLIERS, params=ActionParams(threshold=3.0)),
                Action(action_type=ActionType.MERGE_TABLES, params=ActionParams(merge_key="id")),
                Action(action_type=ActionType.CLEAN_NULLS, params=ActionParams(strategy="fill")),
                Action(action_type=ActionType.VALIDATE, params=ActionParams())
            ]
        
        if self.step < len(actions):
            a = actions[self.step]
            self.step += 1
            return a
        return Action(action_type=ActionType.VALIDATE, params=ActionParams())


def run_episode(env: DataCleanOpsEnv, agent: RLAgent, verbose: bool = True) -> Dict[str, Any]:
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    actions = []
    result = None
    
    if verbose:
        print("\n" + "="*50)
        print(f"Starting Episode - Task: {obs.task}")
        print("="*50)
    
    while not env.done and steps < env.max_steps:
        action = agent.select_action(obs)
        actions.append(action.action_type.value)
        
        if verbose:
            print(f"Step {steps + 1}: {action.action_type.value}")
        
        result = env.step(action)
        
        total_reward += result.reward.value
        steps += 1
        obs = result.observation
        
        if verbose:
            print(f"  Reward: {result.reward.value:+.2f} | Total: {total_reward:+.2f}")
            print(f"  Reason: {result.reward.reason}")
        
        if result.done:
            if verbose:
                print(f"\n*** Episode Terminated ***")
                print(f"Solved: {result.reward.solved}")
            break
    
    if verbose:
        print("\n" + "="*50)
        print("Episode Complete!")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Actions: {actions}")
        print("="*50)
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "actions": actions,
        "solved": result.reward.solved if result and hasattr(result.reward, 'solved') else False
    }


def benchmark():
    print("\n" + "="*50)
    print("DataClean-Ops Agent Benchmark")
    print("="*50)
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n### {task.upper()} ###")
        
        env = create_env(task)
        agent = HeuristicAgent()
        
        result = run_episode(env, agent, verbose=True)
        
        print(f"Total Reward: {result['total_reward']:.2f}")
        print(f"Solved: {result['solved']}")


if __name__ == "__main__":
    print("DataClean-Ops Baseline Testing")
    print("="*50)
    
    env = create_env("easy")
    agent = HeuristicAgent()
    
    result = run_episode(env, agent, verbose=True)
    
    print("\n\nRunning benchmark...")
    benchmark()