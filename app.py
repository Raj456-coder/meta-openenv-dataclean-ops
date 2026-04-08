"""
DataClean-Ops - Hugging Face Spaces
===================================
Simple OpenEnv Interface
"""

import gradio as gr
from env import create_env
from models import Action, ActionType, ActionParams

# Initialize environment
env = create_env("easy")
initial_state = env.reset()

def reset_environment(difficulty):
    """Reset the environment."""
    global env
    env = create_env(difficulty)
    obs = env.reset()
    return f"Reset to {difficulty}. Step: {obs.step}, Tables: {list(obs.tables.keys())}"

def execute_action(action_type):
    """Execute an action in the environment."""
    global env
    
    try:
        action_enum = ActionType(action_type)
    except:
        action_enum = ActionType.VALIDATE
    
    action = Action(action_type=action_enum, params=ActionParams())
    result = env.step(action)
    
    obs = result.observation
    reward = result.reward
    
    output = f"""
Step: {obs.step}
Action: {action_type}
Reward: {reward.value:+.2f}
Total Reward: {obs.reward_accumulated:.2f}
Message: {reward.reason}
Solved: {reward.solved}
Done: {result.done}
Tables: {list(obs.tables.keys())}
"""
    return output

def get_state():
    """Get current state."""
    obs = env.state()
    return f"Task: {obs.task}, Step: {obs.step}/{obs.max_steps}, Reward: {obs.reward_accumulated:.2f}"

# Create Gradio Interface
with gr.Blocks(title="DataClean-Ops") as demo:
    gr.Markdown("# DataClean-Ops: RL Data Engineering Environment")
    gr.Markdown("## Meta OpenEnv Challenge")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Configuration")
            difficulty_dropdown = gr.Dropdown(
                ["easy", "medium", "hard"],
                label="Difficulty",
                value="easy"
            )
            reset_btn = gr.Button("Reset Environment")
        
        with gr.Column():
            gr.Markdown("### Actions")
            action_dropdown = gr.Dropdown(
                ["clean_nulls", "format_date", "drop_duplicates", "merge_tables",
                 "remove_outliers", "normalize_currency", "validate"],
                label="Select Action",
                value="clean_nulls"
            )
            action_btn = gr.Button("Execute Action")
    
    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("### Current State:")
        state_output = gr.Textbox(label="State", value=get_state(), lines=3)
    
    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("### Result:")
        result_output = gr.Textbox(label="Result", lines=6)
    
    # Button clicks
    reset_btn.click(reset_environment, inputs=[difficulty_dropdown], outputs=[state_output])
    action_btn.click(execute_action, inputs=[action_dropdown], outputs=[result_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)