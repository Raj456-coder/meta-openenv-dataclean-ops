"""
DataClean-Ops - HuggingFace Spaces
Simple Flask app with OpenEnv endpoints
"""

import os
import sys
sys.path.insert(0, os.getcwd())

from flask import Flask, jsonify, request, send_from_directory
from env import create_env
from models import Action, ActionType, ActionParams
import json

app = Flask(__name__)
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


@app.route("/reset", methods=["POST"])
def reset():
    global env
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    env = create_env(difficulty)
    obs = env.reset()
    return jsonify({"observation": obs.model_dump(), "done": False})


@app.route("/step", methods=["POST"])
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


@app.route("/validate", methods=["GET"])
def validate():
    global env
    if env is None:
        env = create_env("easy")
    return jsonify(env.validate())


@app.route("/")
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataClean-Ops | OpenEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e8e8e8; min-height: 100vh; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 10px; }
        .badge { background: rgba(0,212,255,0.2); color: #00d4ff; padding: 4px 12px; border-radius: 12px; font-size: 12px; display: inline-block; margin-bottom: 20px; }
        .card { background: #1a1a2e; border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid #2a2a4a; }
        .label { color: #888; font-size: 12px; margin-bottom: 6px; }
        select, button { padding: 12px 16px; border-radius: 8px; border: 1px solid #2a2a4a; background: #16213e; color: #fff; font-size: 14px; margin-right: 8px; }
        button { background: #00d4ff; color: #0f0f1a; cursor: pointer; font-weight: 500; }
        button:hover { background: #00a8cc; }
        #output { background: #0f0f1a; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
        .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
        .action-btn { background: #16213e; border: 1px solid #2a2a4a; padding: 10px 16px; border-radius: 8px; cursor: pointer; }
        .action-btn:hover { border-color: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DataClean-Ops</h1>
        <div class="badge">Meta OpenEnv Challenge</div>
        
        <div class="card">
            <div class="label">Difficulty</div>
            <select id="difficulty">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
            </select>
            <button onclick="resetEnv()">Reset</button>
        </div>
        
        <div class="card">
            <div class="label">Actions</div>
            <div class="actions">
                <button class="action-btn" onclick="execAction('clean_nulls')">clean_nulls</button>
                <button class="action-btn" onclick="execAction('format_date')">format_date</button>
                <button class="action-btn" onclick="execAction('drop_duplicates')">drop_duplicates</button>
                <button class="action-btn" onclick="execAction('merge_tables')">merge_tables</button>
                <button class="action-btn" onclick="execAction('remove_outliers')">remove_outliers</button>
                <button class="action-btn" onclick="execAction('normalize_currency')">normalize_currency</button>
                <button class="action-btn" onclick="execAction('validate')">validate</button>
            </div>
        </div>
        
        <div class="card">
            <div class="label">Output</div>
            <div id="output">Click Reset to start</div>
        </div>
    </div>
    
    <script>
    async function resetEnv() {
        const diff = document.getElementById('difficulty').value;
        const res = await fetch('/reset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({difficulty: diff}) });
        const data = await res.json();
        document.getElementById('output').innerText = JSON.stringify(data, null, 2);
    }
    
    async function execAction(action) {
        const res = await fetch('/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action_type: action, params: {}}) });
        const data = await res.json();
        document.getElementById('output').innerText = JSON.stringify(data, null, 2);
    }
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    print("DataClean-Ops running on http://0.0.0.0:7860")
    app.run(host="0.0.0.0", port=7860)