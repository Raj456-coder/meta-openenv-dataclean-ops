"""
DataClean-Ops V2 - Hugging Face Spaces Deployment
===================================================
Modified version with enhanced features.
"""

import os
import sys
sys.path.insert(0, ".")

from env import create_env
from models import Action, ActionType, ActionParams
import gradio as gr

# Initialize environment
env = create_env("easy")
obs = env.reset()

def predict(action_type, strategy, threshold):
    """Process action and return result."""
    try:
        action = Action(
            action_type=ActionType(action_type),
            params=ActionParams(strategy=strategy, threshold=float(threshold))
        )
        result = env.step(action)
        
        return {
            "step": result.observation.step,
            "reward": result.reward.value,
            "total_reward": result.observation.reward_accumulated,
            "message": result.reward.reason,
            "solved": result.reward.solved,
            "tables": list(result.observation.tables.keys())
        }
    except Exception as e:
        return {"error": str(e)}

def reset(difficulty):
    """Reset environment."""
    global env
    env = create_env(difficulty)
    obs = env.reset()
    return {
        "step": 0,
        "reward": 0.0,
        "total_reward": 0.0,
        "message": f"Environment reset to {difficulty}",
        "solved": False,
        "tables": list(obs.tables.keys())
    }

with gr.Blocks(title="DataClean-Ops V2") as demo:
    gr.Markdown("# DataClean-Ops V2")
    gr.Markdown("Enhanced RL Environment for Data Engineering")
    
    with gr.Row():
        difficulty = gr.Dropdown(["easy", "medium", "hard"], label="Difficulty", value="easy")
        reset_btn = gr.Button("Reset")
    
    with gr.Row():
        action_type = gr.Dropdown(
            ["clean_nulls", "format_date", "drop_duplicates", "merge_tables", "remove_outliers", "normalize_currency", "validate"],
            label="Action"
        )
        strategy = gr.Textbox(value="drop", label="Strategy")
        threshold = gr.Slider(1.0, 5.0, 3.0, label="Threshold")
    
    submit_btn = gr.Button("Execute Action", variant="primary")
    
    output = gr.JSON(label="Result")
    
    reset_btn.click(reset, inputs=[difficulty], outputs=[output])
    submit_btn.click(predict, inputs=[action_type, strategy, threshold], outputs=[output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
