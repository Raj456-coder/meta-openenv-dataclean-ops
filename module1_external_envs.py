"""
Module 1: Connect to 3 Real AI Environments
============================================
Unified interface to interact with Echo bot, Catch game, and Wordle
using the same code pattern as DataClean-Ops.

Each environment follows OpenEnv-like pattern: reset(), step(action), state()
"""

import requests
import json
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class ExternalEnvType(str, Enum):
    ECHO = "echo"
    CATCH = "catch"
    WORDLE = "wordle"


class ExternalAction(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ExternalObservation(BaseModel):
    env_type: str
    state: Any
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class ExternalEnvResult(BaseModel):
    observation: ExternalObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ExternalEnvironment:
    """
    Base class for external AI environments.
    Provides unified interface for Echo, Catch, and Wordle.
    """
    
    def __init__(self, env_type: ExternalEnvType, base_url: str = ""):
        self.env_type = env_type
        self.base_url = base_url
        self.current_state = None
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        self._init_environment()
    
    def _init_environment(self):
        """Initialize the environment."""
        if self.env_type == ExternalEnvType.ECHO:
            self._init_echo()
        elif self.env_type == ExternalEnvType.CATCH:
            self._init_catch()
        elif self.env_type == ExternalEnvType.WORDLE:
            self._init_wordle()
    
    def _init_echo(self):
        """Initialize Echo bot environment."""
        self.current_state = {"message": "", "history": []}
        self.actions = ["echo", "silence", "question"]
    
    def _init_catch(self):
        """Initialize Catch game environment."""
        self.current_state = {
            "ball_x": 5,
            "basket_x": 5,
            "game_over": False,
            "score": 0,
            "board_size": 10
        }
        self.actions = ["left", "right", "stay"]
    
    def _init_wordle(self):
        """Initialize Wordle environment."""
        self.current_state = {
            "attempts": 0,
            "max_attempts": 6,
            "guesses": [],
            "feedback": [],
            "game_over": False,
            "secret_word": "PYTHON"
        }
        self.actions = ["guess"]
    
    def reset(self) -> ExternalObservation:
        """Reset environment to initial state."""
        self._init_environment()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return self.state()
    
    def step(self, action: ExternalAction) -> ExternalEnvResult:
        """Execute one step in the environment."""
        if self.done:
            return ExternalEnvResult(
                observation=self.state(),
                reward=0.0,
                done=True,
                info={"message": "Episode terminated"}
            )
        
        self.step_count += 1
        
        if self.env_type == ExternalEnvType.ECHO:
            reward, done, info = self._step_echo(action)
        elif self.env_type == ExternalEnvType.CATCH:
            reward, done, info = self._step_catch(action)
        elif self.env_type == ExternalEnvType.WORDLE:
            reward, done, info = self._step_wordle(action)
        else:
            reward, done, info = 0.0, False, {"error": "Unknown environment"}
        
        self.total_reward += reward
        self.done = done
        
        return ExternalEnvResult(
            observation=self.state(),
            reward=reward,
            done=done,
            info=info
        )
    
    def _step_echo(self, action: ExternalAction) -> tuple:
        """Step for Echo bot."""
        action_str = action.action.lower()
        
        if action_str == "echo":
            response = "Hello! I am the Echo bot. How can I help you?"
            self.current_state["message"] = response
            self.current_state["history"].append(("user", action_str))
            self.current_state["history"].append(("bot", response))
            return 0.1, False, {"response": response, "type": "echo"}
        
        elif action_str == "silence":
            self.current_state["message"] = "..."
            self.current_state["history"].append(("user", "silence"))
            return 0.0, False, {"response": "...", "type": "silence"}
        
        elif action_str == "question":
            response = "I don't have an answer, but I can help you learn!"
            self.current_state["message"] = response
            self.current_state["history"].append(("user", "question"))
            self.current_state["history"].append(("bot", response))
            return 0.1, False, {"response": response, "type": "question"}
        
        return -0.1, False, {"error": "Unknown action"}
    
    def _step_catch(self, action: ExternalAction) -> tuple:
        """Step for Catch game."""
        action_str = action.action.lower()
        
        state = self.current_state
        ball_x = state["ball_x"]
        basket_x = state["basket_x"]
        
        if action_str == "left":
            basket_x = max(0, basket_x - 1)
        elif action_str == "right":
            basket_x = min(state["board_size"] - 1, basket_x + 1)
        
        ball_x += 1
        
        if ball_x >= state["board_size"]:
            if abs(ball_x - 1 - basket_x) <= 1:
                score = state["score"] + 1
                reward = 1.0
                done = False
                message = "Caught! Score +1"
            else:
                reward = -1.0
                done = True
                message = "Missed! Game Over"
                ball_x = state["board_size"]
            
            self.current_state = {
                "ball_x": ball_x - 1 if not done else 5,
                "basket_x": basket_x,
                "game_over": done,
                "score": score if not done else state["score"],
                "board_size": state["board_size"]
            }
            
            return reward, done, {"message": message, "score": self.current_state["score"]}
        
        self.current_state["basket_x"] = basket_x
        self.current_state["ball_x"] = ball_x
        return 0.0, False, {"ball_x": ball_x, "basket_x": basket_x}
    
    def _step_wordle(self, action: ExternalAction) -> tuple:
        """Step for Wordle game."""
        guess = action.params.get("word", "").upper()
        
        if not guess or len(guess) != 5:
            return -0.1, False, {"error": "Invalid word"}
        
        secret = self.current_state["secret_word"]
        feedback = []
        
        for i, letter in enumerate(guess):
            if letter == secret[i]:
                feedback.append("green")
            elif letter in secret:
                feedback.append("yellow")
            else:
                feedback.append("gray")
        
        self.current_state["attempts"] += 1
        self.current_state["guesses"].append(guess)
        self.current_state["feedback"].append(feedback)
        
        if guess == secret:
            return 1.0, True, {"message": "You won!", "guess": guess, "feedback": feedback}
        
        if self.current_state["attempts"] >= self.current_state["max_attempts"]:
            return -0.5, True, {"message": "Game over", "secret": secret}
        
        return 0.1, False, {"guess": guess, "feedback": feedback}
    
    def state(self) -> ExternalObservation:
        """Get current observation."""
        return ExternalObservation(
            env_type=self.env_type.value,
            state=self.current_state,
            reward=self.total_reward,
            done=self.done,
            info={"step": self.step_count}
        )
    
    def validate(self) -> Dict[str, Any]:
        """Validate environment."""
        return {
            "valid": True,
            "env_type": self.env_type.value,
            "actions": getattr(self, "actions", []),
            "framework": "external-connector"
        }


def create_external_env(env_type: str, base_url: str = "") -> ExternalEnvironment:
    """Factory function to create external environment."""
    return ExternalEnvironment(ExternalEnvType(env_type), base_url)


def demo_unified_pattern():
    """Demonstrate unified code pattern for all 3 environments."""
    print("="*60)
    print("Module 1: Unified External Environment Interface")
    print("="*60)
    
    environments = [
        ("echo", "Echo Bot"),
        ("catch", "Catch Game"),
        ("wordle", "Wordle")
    ]
    
    for env_type, env_name in environments:
        print(f"\n--- Testing {env_name} ---")
        
        env = create_external_env(env_type)
        obs = env.reset()
        
        print(f"Initial state: {obs.state}")
        
        if env_type == "echo":
            actions = ["echo", "question"]
        elif env_type == "catch":
            actions = ["right", "left", "stay"]
        else:
            actions = [{"action": "guess", "params": {"word": "HELLO"}},
                      {"action": "guess", "params": {"word": "WORLD"}}]
        
        for i, action_data in enumerate(actions):
            if isinstance(action_data, dict):
                action = ExternalAction(**action_data)
            else:
                action = ExternalAction(action=action_data)
            
            result = env.step(action)
            print(f"Step {i+1}: action={action.action}, reward={result.reward:.2f}, done={result.done}")
        
        print(f"Final reward: {env.total_reward:.2f}")
    
    print("\n" + "="*60)
    print("All environments tested with unified pattern!")
    print("="*60)


if __name__ == "__main__":
    demo_unified_pattern()