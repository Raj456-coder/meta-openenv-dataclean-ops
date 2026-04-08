"""
DataClean-Ops - Hugging Face Spaces Deployment
=============================================
RL Environment for Data Engineering
"""

import os
import sys
sys.path.insert(0, ".")

from env import create_env
from models import Action, ActionType, ActionParams
import gradio as gr

# Global environment instance
_env = None

def get_env():
    global _env
    if _env is None:
        _env = create_env("easy")
        _env.reset()
    return _env

def reset_env(difficulty):
    """Reset environment to initial state."""
    global _env
    _env = create_env(difficulty)
    obs = _env.reset()
    return {
        "status": "reset",
        "task": obs.task,
        "step": obs.step,
        "max_steps": obs.max_steps,
        "tables": list(obs.tables.keys()),
        "reward": 0.0,
        "message": f"Environment reset to {difficulty}"
    }

def execute_action(action_type, strategy, threshold):
    """Execute an action and return result."""
    env = get_env()
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
            "done": result.done,
            "tables": list(result.observation.tables.keys())
        }
    except Exception as e:
        return {"error": str(e), "step": env.step_count}

def get_state():
    """Get current environment state."""
    env = get_env()
    obs = env.state()
    return {
        "task": obs.task,
        "step": obs.step,
        "max_steps": obs.max_steps,
        "tables": list(obs.tables.keys()),
        "reward_accumulated": obs.reward_accumulated,
        "action_history": obs.action_history
    }

# Build Gradio Interface
with gr.Blocks(title="DataClean-Ops - RL Environment") as demo:
    gr.Markdown("# DataClean-Ops: RL Environment for Data Engineering")
    gr.Markdown("## OpenEnv Compliant - Step/Reset/State API")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Configuration")
            difficulty = gr.Dropdown(
                ["easy", "medium", "hard"], 
                label="Task Difficulty", 
                value="easy"
            )
            reset_btn = gr.Button("Reset Environment", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### Actions")
            action_type = gr.Dropdown(
                ["clean_nulls", "format_date", "drop_duplicates", "merge_tables", 
                 "remove_outliers", "normalize_currency", "validate"],
                label="Select Action",
                value="clean_nulls"
            )
            strategy = gr.Textbox(value="drop", label="Strategy (drop/fill)")
            threshold = gr.Slider(1.0, 5.0, 3.0, label="Outlier Threshold")
            submit_btn = gr.Button("Execute Action", variant="primary")
    
    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("### Result Output")
        output = gr.JSON(label="Execution Result")
    
    # Event handlers
    reset_btn.click(reset_env, inputs=[difficulty], outputs=[output])
    submit_btn.click(execute_action, inputs=[action_type, strategy, threshold], outputs=[output])
    
    gr.Markdown("---")
    gr.Markdown("*Built for Meta OpenEnv Challenge - Data Engineering Task*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)