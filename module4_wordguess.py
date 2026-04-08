"""
Module 4: Word Guessing Game from Scratch
===========================================
Build complete environment, test locally, deploy to HF Spaces.
Approximately 100 lines of real code.
"""

import random
import string
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class WordGuessAction(BaseModel):
    """Action for word guessing game."""
    guess: str


class WordGuessObservation(BaseModel):
    """Observation state."""
    attempts: int
    max_attempts: int
    feedback: List[List[str]] = Field(default_factory=list)
    game_over: bool
    solved: bool


class WordGuessEnvironment:
    """Complete word guessing game environment."""
    
    WORD_LIST = [
        "PYTHON", "CODE", "DATA", "GAME", "LEARN",
        "ROBOT", "AI", "OPEN", "ENV", "MODEL",
        "TRAIN", "SMART", "FAST", "TECH", "BYTE",
        "LOGIC", "NEURAL", "POWER", "SPACE", "CLOUD"
    ]
    
    def __init__(self, word: Optional[str] = None, max_attempts: int = 6):
        self.secret_word = word or random.choice(self.WORD_LIST)
        self.max_attempts = max_attempts
        self.attempts = 0
        self.guesses: List[str] = []
        self.feedback: List[List[str]] = []
        self.game_over = False
        self.solved = False
    
    def reset(self) -> WordGuessObservation:
        """Reset game to initial state."""
        self.secret_word = random.choice(self.WORD_LIST)
        self.attempts = 0
        self.guesses = []
        self.feedback = []
        self.game_over = False
        self.solved = False
        return self._get_obs()
    
    def step(self, action: WordGuessAction) -> Tuple[WordGuessObservation, float, bool]:
        """Execute one guess."""
        if self.game_over:
            return self._get_obs(), 0.0, True
        
        guess = action.guess.upper()
        
        if len(guess) != 5 or not guess.isalpha():
            self.attempts += 1
            self.game_over = self.attempts >= self.max_attempts
            self.feedback.append(["invalid"] * 5)
            return self._get_obs(), -0.1, self.game_over
        
        if guess == self.secret_word:
            self.solved = True
            self.game_over = True
            return self._get_obs(), 1.0, True
        
        fb = self._get_feedback(guess)
        self.feedback.append(fb)
        self.attempts += 1
        self.game_over = self.attempts >= self.max_attempts
        
        reward = 0.1 if self.attempts < self.max_attempts else -0.5
        return self._get_obs(), reward, self.game_over
    
    def _get_feedback(self, guess: str) -> List[str]:
        """Generate feedback: green=exact, yellow=present, gray=absent."""
        feedback = []
        for i, letter in enumerate(guess):
            if i < len(self.secret_word) and letter == self.secret_word[i]:
                feedback.append("green")
            elif letter in self.secret_word:
                feedback.append("yellow")
            else:
                feedback.append("gray")
        return feedback
    
    def _get_obs(self) -> WordGuessObservation:
        return WordGuessObservation(
            attempts=self.attempts,
            max_attempts=self.max_attempts,
            feedback=self.feedback.copy(),
            game_over=self.game_over,
            solved=self.solved
        )
    
    def state(self) -> WordGuessObservation:
        return self._get_obs()
    
    def validate(self) -> Dict[str, Any]:
        return {"valid": True, "framework": "wordguess-v1", "actions": ["guess"]}


def test_local():
    """Test the environment locally."""
    print("="*50)
    print("Testing Word Guess Environment Locally")
    print("="*50)
    
    env = WordGuessEnvironment()
    obs = env.reset()
    
    print(f"\nNew Game! Word: {env.secret_word}")
    print(f"Attempts: {obs.attempts}/{obs.max_attempts}")
    
    test_guesses = ["HELLO", "WORLD", "CODE", "SMART", "TRAIN", "PYTHON"]
    
    for guess in test_guesses:
        obs, reward, done = env.step(WordGuessAction(guess=guess))
        print(f"Guess: {guess:5} | Reward: {reward:+.1f} | Done: {done}")
        
        if done:
            print(f"\n{'WON!' if obs.solved else 'LOST!'} Final attempts: {obs.attempts}")
            break
    
    print(f"\nValidation: {env.validate()}")
    return obs.solved


def deploy_to_hf():
    """Generate deployment files for HF Spaces."""
    print("\n" + "="*50)
    print("Creating HF Spaces Deployment Files")
    print("="*50)
    
    app_code = '''import gradio as gr
from module4_wordguess import WordGuessEnvironment

env = WordGuessEnvironment()

def play(guess):
    obs, reward, done = env.step(WordGuessAction(guess=guess.upper()))
    if done:
        result = f"Game Over! {'WON' if obs.solved else 'LOST'}"
        env.reset()
    else:
        fb = obs.feedback[-1] if obs.feedback else []
        result = f"Attempts: {obs.attempts}/{obs.max_attempts} | Feedback: {fb}"
    return result

demo = gr.Interface(play, "text", "text", title="Word Guess Game")
demo.launch(server_name="0.0.0.0", server_port=7860)
'''
    
    with open("app.py", "w") as f:
        f.write(app_code)
    print("  Created: app.py")
    
    with open("requirements.txt", "w") as f:
        f.write("gradio>=4.0.0\npydantic>=2.0.0\n")
    print("  Created: requirements.txt")
    
    print("\n[OK] Ready for deployment!")
    print("  Run: python app.py  # for local test")
    print("  Push to HF Spaces for live deployment")


if __name__ == "__main__":
    solved = test_local()
    deploy_to_hf()