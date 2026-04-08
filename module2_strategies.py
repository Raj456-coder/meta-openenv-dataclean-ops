"""
Module 2: Game-Playing Strategies for Catch Game
=================================================
Implements 4 different strategies, runs competition, 
then switches to a different game using same code pattern.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class GameStrategy(ABC):
    """Abstract base class for game strategies."""
    
    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> str:
        """Given game state, return the best action."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset strategy state."""
        pass


class RandomStrategy(GameStrategy):
    """Strategy 1: Random moves"""
    
    def __init__(self, actions: List[str]):
        self.actions = actions
    
    def get_action(self, state: Dict[str, Any]) -> str:
        return random.choice(self.actions)
    
    def reset(self):
        pass


class FollowBallStrategy(GameStrategy):
    """Strategy 2: Always move towards the ball"""
    
    def get_action(self, state: Dict[str, Any]) -> str:
        ball_x = state.get("ball_x", 5)
        basket_x = state.get("basket_x", 5)
        
        if ball_x < basket_x:
            return "left"
        elif ball_x > basket_x:
            return "right"
        return "stay"
    
    def reset(self):
        pass


class PredictStrategy(GameStrategy):
    """Strategy 3: Predict ball landing position"""
    
    def __init__(self):
        self.predictions = []
    
    def get_action(self, state: Dict[str, Any]) -> str:
        ball_x = state.get("ball_x", 5)
        basket_x = state.get("basket_x", 5)
        board_size = state.get("board_size", 10)
        
        steps_to_land = board_size - ball_x - 1
        predicted_x = basket_x
        
        for _ in range(steps_to_land):
            predicted_x = max(0, min(board_size - 1, predicted_x + random.choice([-1, 0, 1])))
        
        if predicted_x < basket_x:
            return "left"
        elif predicted_x > basket_x:
            return "right"
        return "stay"
    
    def reset(self):
        self.predictions = []


class CenterBiasStrategy(GameStrategy):
    """Strategy 4: Prefer center position, react only when ball is close"""
    
    def __init__(self):
        self.center = 5
    
    def get_action(self, state: Dict[str, Any]) -> str:
        ball_x = state.get("ball_x", 5)
        basket_x = state.get("basket_x", 5)
        
        if ball_x > 7 and basket_x < self.center:
            return "right"
        elif ball_x > 7 and basket_x > self.center:
            return "left"
        elif ball_x < 3 and basket_x > self.center:
            return "left"
        elif ball_x < 3 and basket_x < self.center:
            return "right"
        return "stay"
    
    def reset(self):
        pass


class CatchGame:
    """Catch game environment."""
    
    def __init__(self):
        self.ball_x = 5
        self.basket_x = 5
        self.score = 0
        self.board_size = 10
        self.done = False
    
    def reset(self) -> Dict[str, Any]:
        self.ball_x = random.randint(0, self.board_size - 1)
        self.basket_x = self.board_size // 2
        self.score = 0
        self.done = False
        return self.get_state()
    
    def step(self, action: str) -> tuple:
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        if action == "left":
            self.basket_x = max(0, self.basket_x - 1)
        elif action == "right":
            self.basket_x = min(self.board_size - 1, self.basket_x + 1)
        
        self.ball_x += 1
        
        if self.ball_x >= self.board_size:
            if abs(self.ball_x - 1 - self.basket_x) <= 1:
                self.score += 1
                reward = 1.0
                message = "Caught!"
            else:
                self.done = True
                reward = -1.0
                message = "Missed!"
            
            self.ball_x = random.randint(0, self.board_size - 1)
            self.basket_x = self.board_size // 2
            
            return self.get_state(), reward, self.done, {"message": message, "score": self.score}
        
        return self.get_state(), 0.0, False, {}
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "ball_x": self.ball_x,
            "basket_x": self.basket_x,
            "score": self.score,
            "board_size": self.board_size,
            "game_over": self.done
        }


class SnakeGame:
    """Snake game - different game using same pattern."""
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.snake = [(5, 5), (5, 6), (5, 7)]
        self.food = (random.randint(0, width-1), random.randint(0, height-1))
        self.direction = (0, -1)
        self.score = 0
        self.done = False
    
    def reset(self) -> Dict[str, Any]:
        self.snake = [(5, 5), (5, 6), (5, 7)]
        self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
        self.direction = (0, -1)
        self.score = 0
        self.done = False
        return self.get_state()
    
    def step(self, action: str) -> tuple:
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        dir_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        
        if action in dir_map:
            new_dir = dir_map[action]
            if (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) != (0, 0):
                self.direction = new_dir
        
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.done = True
            return self.get_state(), -1.0, True, {"message": "Game Over"}
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            reward = 1.0
        else:
            self.snake.pop()
            reward = 0.0
        
        return self.get_state(), reward, self.done, {"score": self.score}
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "snake": self.snake,
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
            "game_over": self.done
        }


def run_competition():
    """Run competition between strategies on Catch game."""
    print("="*60)
    print("Module 2: Catch Game Strategy Competition")
    print("="*60)
    
    strategies = [
        ("Random", RandomStrategy(["left", "right", "stay"])),
        ("Follow Ball", FollowBallStrategy()),
        ("Predict", PredictStrategy()),
        ("Center Bias", CenterBiasStrategy())
    ]
    
    results = []
    
    for name, strategy in strategies:
        print(f"\n--- Testing: {name} ---")
        
        game = CatchGame()
        state = game.reset()
        total_reward = 0.0
        episodes = 5
        episode_scores = []
        
        for ep in range(episodes):
            ep_reward = 0.0
            steps = 0
            
            while not game.done and steps < 50:
                action = strategy.get_action(state)
                state, reward, done, info = game.step(action)
                ep_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_scores.append(game.score)
            total_reward += ep_reward
            strategy.reset()
            
            if ep < episodes - 1:
                game.reset()
        
        avg_score = np.mean(episode_scores)
        results.append((name, avg_score, total_reward))
        print(f"  Avg Score: {avg_score:.2f}, Total Reward: {total_reward:.2f}")
    
    print("\n" + "="*60)
    print("LEADERBOARD:")
    print("="*60)
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (name, score, reward) in enumerate(results, 1):
        print(f"{i}. {name}: Avg Score = {score:.2f}")
    
    return results


def switch_to_snake():
    """Switch to Snake game using same code pattern."""
    print("\n" + "="*60)
    print("Switching to Snake Game (Same Code Pattern)")
    print("="*60)
    
    game = SnakeGame()
    strategy = FollowBallStrategy()
    
    state = game.reset()
    print(f"Initial state: score={state['score']}, snake_len={len(state['snake'])}")
    
    actions = ["up", "down", "left", "right", "up", "right", "down"]
    
    for i, action in enumerate(actions):
        state, reward, done, info = game.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, score={info.get('score', 0)}, done={done}")
        
        if done:
            print(f"Game Over! Final Score: {game.score}")
            break
    
    print("\nDemonstrated: Same code pattern works for different games!")
    return game.score


if __name__ == "__main__":
    results = run_competition()
    final_score = switch_to_snake()
    print(f"\nFinal Snake Score: {final_score}")