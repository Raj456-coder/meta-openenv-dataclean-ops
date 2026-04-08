"""
DataClean-Ops - HuggingFace Spaces
Simple Gradio Interface for OpenEnv
"""

import gradio as gr
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Import our modules
from env import create_env
from models import Action, ActionType, ActionParams

# Initialize
env = create_env("easy")
_ = env.reset()

def reset_env(difficulty):
    """Reset environment."""
    global env
    env = create_env(difficulty)
    obs = env.reset()
    return f"Reset to {difficulty}\nStep: {obs.step}\nTables: {list(obs.tables.keys())}"

def run_action(action):
    """Execute action."""
    global env
    try:
        action_type = ActionType(action)
    except:
        action_type = ActionType.VALIDATE
    
    act = Action(action_type=action_type, params=ActionParams())
    result = env.step(act)
    
    obs = result.observation
    rew = result.reward
    
    return f"Action: {action}\nReward: {rew.value:+.2f}\nTotal: {obs.reward_accumulated:.2f}\nStep: {obs.step}\n{rew.reason}"

def get_current_state():
    """Get current state."""
    global env
    obs = env.state()
    return f"Task: {obs.task}\nStep: {obs.step}/{obs.max_steps}\nReward: {obs.reward_accumulated:.2f}\nTables: {list(obs.tables.keys())}"

# Build Gradio Interface
demo = gr.Blocks(title="DataClean-Ops")

with demo:
    gr.Markdown("# DataClean-Ops: RL Data Engineering")
    gr.Markdown("### Meta OpenEnv Challenge")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Setup")
            diff = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Difficulty")
            btn_reset = gr.Button("Reset")
        
        with gr.Column():
            gr.Markdown("### Action")
            action = gr.Dropdown(
                ["clean_nulls", "format_date", "drop_duplicates", "merge_tables", 
                 "remove_outliers", "normalize_currency", "validate"],
                label="Action",
                value="clean_nulls"
            )
            btn_run = gr.Button("Execute")
    
    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("### State:")
        state = gr.Textbox(value=get_current_state, lines=3, label="Current State")
    
    gr.Markdown("---")
    
    gr.Markdown("### Result:")
    result = gr.Textbox(lines=4, label="Output")
    
    # Events
    btn_reset.click(reset_env, inputs=[diff], outputs=[state])
    btn_run.click(run_action, inputs=[action], outputs=[result])

# Launch
demo.launch(server_name="0.0.0.0", server_port=7860)