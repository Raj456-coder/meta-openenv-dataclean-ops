"""
DataClean-Ops - HuggingFace Spaces
Simple Fixed Version
"""

import gradio as gr
import os
import sys

sys.path.insert(0, os.getcwd())

from env import create_env
from models import Action, ActionType, ActionParams

def run_action(difficulty, action_name):
    try:
        env = create_env(difficulty)
        obs = env.reset()
        
        if action_name and action_name != "None":
            at = ActionType(action_name)
            action = Action(action_type=at, params=ActionParams())
            result = env.step(action)
            return f"Step: {result.observation.step}\nReward: {result.reward.value:.2f}\nDone: {result.done}"
        
        return f"Tables: {list(obs.tables.keys())}"
    except Exception as e:
        return f"Error: {str(e)}"

def reset_env(difficulty):
    try:
        env = create_env(difficulty)
        obs = env.reset()
        return f"Reset done\nTables: {list(obs.tables.keys())}"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# DataClean-Ops: RL Environment")
    gr.Markdown("### Meta OpenEnv Challenge")
    
    with gr.Row():
        diff = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Difficulty")
    
    with gr.Row():
        reset_btn = gr.Button("Reset")
        run_btn = gr.Button("Execute")
    
    action_dropdown = gr.Dropdown(
        ["clean_nulls", "format_date", "drop_duplicates", "merge_tables", 
         "remove_outliers", "normalize_currency", "validate"],
        label="Action",
        value="clean_nulls"
    )
    
    output = gr.Textbox(lines=5, label="Result")
    
    reset_btn.click(reset_env, inputs=[diff], outputs=[output])
    run_btn.click(run_action, inputs=[diff, action_dropdown], outputs=[output])

demo.launch(server_name="0.0.0.0", server_port=7860)
