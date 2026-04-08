"""
DataClean-Ops - Unified Gradio + Flask for OpenEnv
Serves both UI (Gradio) and API endpoints (/reset, /step)
"""

import gradio as gr
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

sys.path.insert(0, os.getcwd())

from env import create_env
from models import Action, ActionType, ActionParams

# Flask API setup
flask_app = Flask(__name__)
CORS(flask_app)

# Global environment
env = None


def action_from_dict(data):
    at = data.get("action_type", "validate")
    params_data = data.get("params", {})
    
    params = ActionParams(
        columns=params_data.get("columns"),
        strategy=params_data.get("strategy", "default"),
        threshold=params_data.get("threshold", 3.0)
    )
    
    for action_type in ActionType:
        if action_type.value == at:
            return Action(type=action_type, params=params)
    
    return Action(type=ActionType.VALIDATE, params=params)


@flask_app.route("/reset", methods=["POST"])
def reset():
    global env
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    env = create_env(difficulty)
    obs = env.reset()
    return jsonify({"observation": obs.model_dump(), "done": False})


@flask_app.route("/step", methods=["POST"])
def step():
    global env
    if env is None:
        env = create_env("easy")
        env.reset()
    
    data = request.json or {}
    action = action_from_dict(data)
    result = env.step(action)
    
    return jsonify({
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done
    })


@flask_app.route("/validate", methods=["GET"])
def validate():
    global env
    if env is None:
        env = create_env("easy")
    return jsonify(env.validate())


# Gradio UI
def get_result(difficulty, action_name):
    global env
    try:
        if env is None:
            env = create_env(difficulty)
            env.reset()
        
        if action_name and action_name != "None":
            at = ActionType(action_name)
            action = Action(action_type=at, params=ActionParams())
            result = env.step(action)
            return f"Step: {env.step_count}\nReward: {result.reward.value:.2f}\nDone: {result.done}\nSolved: {result.reward.solved}"
        
        tables = env.data if hasattr(env, 'data') else {}
        return f"Tables: {list(tables.keys())}"
    except Exception as e:
        return f"Error: {str(e)}"


def reset_env(difficulty):
    global env
    try:
        env = create_env(difficulty)
        obs = env.reset()
        return f"Reset done\nTables: {list(obs.tables.keys())}\nTask: {obs.task}"
    except Exception as e:
        return f"Error: {str(e)}"


# Mount Flask in Gradio
app = gr.mount_gradio_app(flask_app, gr.Blocks(title="DataClean-Ops"), path="/")

# Add root route for Gradio
@flask_app.route("/")
def index():
    return app.launch()


if __name__ == "__main__":
    print("=" * 50)
    print("DataClean-Ops - OpenEnv RL Environment")
    print("=" * 50)
    print("API Endpoints:")
    print("  POST /reset   - Reset environment")
    print("  POST /step    - Execute action")
    print("  GET  /validate - Validate state")
    print("=" * 50)
    flask_app.run(host="0.0.0.0", port=7860)