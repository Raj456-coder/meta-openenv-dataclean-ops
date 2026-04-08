from flask import Flask, jsonify, request
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


@app.route("/")
def index():
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataClean-Ops | RL Data Engineering</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --accent: #00d4ff;
            --accent-hover: #00a8cc;
            --success: #00ff88;
            --warning: #ffaa00;
            --danger: #ff4757;
            --text: #e8e8e8;
            --text-muted: #888;
            --border: #2a2a4a;
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text);
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            padding: 20px 40px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: -0.5px;
        }
        
        .logo span { color: #fff; }
        
        .badge {
            background: rgba(0, 212, 255, 0.1);
            color: var(--accent);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 40px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 14px;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--accent);
            color: var(--bg-primary);
        }
        
        .btn-primary:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent);
        }
        
        .form-group {
            margin-bottom: 16px;
        }
        
        .form-label {
            display: block;
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        
        .form-select, .form-input {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        .form-select:focus, .form-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .action-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        
        .action-btn {
            padding: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        
        .action-btn:hover {
            border-color: var(--accent);
            background: rgba(0, 212, 255, 0.1);
        }
        
        .action-btn.active {
            border-color: var(--accent);
            background: rgba(0, 212, 255, 0.2);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        
        .stat-box {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
        }
        
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.danger { color: var(--danger); }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        
        .table-data {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        
        .table-data th {
            text-align: left;
            padding: 12px;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
        }
        
        .table-data td {
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .table-data tr:hover td {
            background: rgba(255,255,255,0.02);
        }
        
        .code-block {
            background: var(--bg-primary);
            padding: 16px;
            border-radius: 8px;
            font-family: 'Monaco', monospace;
            font-size: 12px;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .history-item {
            padding: 8px 12px;
            background: var(--bg-secondary);
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }
        
        .history-reward {
            font-weight: 600;
        }
        
        .history-reward.positive { color: var(--success); }
        .history-reward.negative { color: var(--danger); }
        
        .full-width { grid-column: 1 / -1; }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .action-grid { grid-template-columns: repeat(2, 1fr); }
            .container { padding: 20px; }
            .header { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">DataClean<span>-Ops</span></div>
        <div class="badge">OpenEnv v1.0.0</div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Configuration</span>
                </div>
                <div class="form-group">
                    <label class="form-label">Task Difficulty</label>
                    <select class="form-select" id="difficulty">
                        <option value="easy">Easy - Missing Values</option>
                        <option value="medium">Medium - Duplicates</option>
                        <option value="hard">Hard - Outlier Detection</option>
                    </select>
                </div>
                <button class="btn btn-primary" onclick="resetEnv()">Reset Environment</button>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Execute Action</span>
                </div>
                <div class="action-grid">
                    <div class="action-btn active" onclick="selectAction('drop_nulls', this)">drop_nulls</div>
                    <div class="action-btn" onclick="selectAction('fill_value', this)">fill_value</div>
                    <div class="action-btn" onclick="selectAction('remove_duplicates', this)">rem_dupes</div>
                    <div class="action-btn" onclick="selectAction('outlier_remove', this)">outlier</div>
                    <div class="action-btn" onclick="selectAction('normalize_column', this)">normalize</div>
                    <div class="action-btn" onclick="selectAction('validate', this)">validate</div>
                </div>
                <button class="btn btn-secondary" style="margin-top: 16px; width: 100%;" onclick="executeAction()">Execute Selected Action</button>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Episode Metrics</span>
                </div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="step">0</div>
                        <div class="stat-label">Current Step</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="maxSteps">20</div>
                        <div class="stat-label">Max Steps</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="reward">0.00</div>
                        <div class="stat-label">Total Reward</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="status">Ready</div>
                        <div class="stat-label">Status</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Action History</span>
                </div>
                <div id="history">
                    <div class="history-item">
                        <span>No actions yet</span>
                    </div>
                </div>
            </div>
            
            <div class="card full-width">
                <div class="card-header">
                    <span class="card-title">Data State - Tables</span>
                </div>
                <div id="tables">
                    <table class="table-data">
                        <thead>
                            <tr>
                                <th>Table</th>
                                <th>Rows</th>
                                <th>Columns</th>
                                <th>Null %</th>
                                <th>Duplicates</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                            <tr><td colspan="5">No data</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card full-width">
                <div class="card-header">
                    <span class="card-title">Raw JSON Output</span>
                </div>
                <div class="code-block" id="jsonOutput">
                    {'status': 'Click Reset to start'}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedAction = 'drop_nulls';
        
        function selectAction(action, element) {
            document.querySelectorAll('.action-btn').forEach(btn => btn.classList.remove('active'));
            element.classList.add('active');
            selectedAction = action;
        }
        
        async function resetEnv() {
            const diff = document.getElementById('difficulty').value;
            const res = await fetch('/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({difficulty: diff})
            });
            const data = await res.json();
            updateUI(data);
        }
        
        async function executeAction() {
            const res = await fetch('/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    action_type: selectedAction,
                    params: {threshold: 3.0}
                })
            });
            const data = await res.json();
            updateUI(data);
        }
        
        function updateUI(data) {
            const obs = data.observation;
            const rew = data.reward || {value: 0, reason: ''};
            
            document.getElementById('step').innerText = obs.step;
            document.getElementById('maxSteps').innerText = obs.max_steps;
            document.getElementById('reward').innerText = obs.reward_accumulated.toFixed(2);
            document.getElementById('reward').className = 'stat-value ' + (obs.reward_accumulated >= 0 ? 'success' : 'danger');
            
            const statusEl = document.getElementById('status');
            if (data.done) {
                statusEl.innerText = rew.solved ? 'SOLVED!' : 'Done';
                statusEl.className = 'stat-value ' + (rew.solved ? 'success' : 'warning');
            } else {
                statusEl.innerText = 'Running';
                statusEl.className = 'stat-value';
            }
            
            let html = '';
            obs.action_history.forEach((action, i) => {
                html += '<div class="history-item">';
                html += '<span>' + (i+1) + '. ' + action + '</span>';
                html += '</div>';
            });
            document.getElementById('history').innerHTML = html || '<div class="history-item"><span>No actions yet</span></div>';
            
            let tableHtml = '';
            for (const [name, table] of Object.entries(obs.tables)) {
                tableHtml += '<tr>';
                tableHtml += '<td>' + name + '</td>';
                tableHtml += '<td>' + table.rows + '</td>';
                tableHtml += '<td>' + table.cols + '</td>';
                tableHtml += '<td>' + table.null_pct.toFixed(1) + '%</td>';
                tableHtml += '<td>' + table.dupes + '</td>';
                tableHtml += '</tr>';
            }
            document.getElementById('tableBody').innerHTML = tableHtml || '<tr><td colspan="5">No data</td></tr>';
            
            document.getElementById('jsonOutput').innerText = JSON.stringify(data, null, 2);
        }
        
        resetEnv();
    </script>
</body>
</html>'''


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
        env = create_env()
        env.reset()
    
    data = request.json or {}
    action = action_from_dict(data)
    result = env.step(action)
    
    return jsonify({
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done
    })


@app.route("/validate")
def validate():
    global env
    if env is None:
        env = create_env()
    return jsonify(env.validate())


if __name__ == "__main__":
    print("="*50)
    print("DataClean-Ops SaaS Interface")
    print("="*50)
    print("Running at: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)