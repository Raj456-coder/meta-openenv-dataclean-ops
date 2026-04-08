# DataClean-Ops: Data Engineering Training Environment

A professional reinforcement learning environment for training AI agents to clean, transform, and validate messy corporate datasets through a standard OpenEnv API.

## Overview

DataClean-Ops simulates real-world data engineering tasks where an AI agent learns to:
- Handle missing values in datasets
- Normalize date and currency formats
- Detect and remove statistical outliers
- Merge mismatched datasets

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

```
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
rich>=13.0.0
openai>=1.0.0
```

## Quick Start

```python
from env import create_env
from models import Action, ActionType, ActionParams
from baseline import HeuristicAgent, run_episode

# Create environment
env = create_env("easy")

# Reset and get initial observation
obs = env.reset()

# Create agent and run episode
agent = HeuristicAgent()
result = run_episode(env, agent, verbose=True)
```

## Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `clean_nulls` | Handle missing values | `strategy`: "drop" or "fill", `target_columns`: List[str] |
| `format_date` | Normalize date formats | `target_columns`: List[str] |
| `drop_duplicates` | Remove duplicate rows | - |
| `merge_tables` | Merge multiple tables | `merge_key`: str |
| `remove_outliers` | Remove statistical outliers | `threshold`: float (default 3.0) |
| `normalize_currency` | Normalize currency values | - |
| `validate_schema` | Validate and grade solution | - |
| `no_op` | No operation | - |

## Observation Space

### DataState
- **tables**: Dict[str, TableSnapshot] - Dictionary of table snapshots
- **active_table**: Optional[str] - Currently active table name
- **task_type**: str - Current task difficulty ("easy", "medium", "hard")
- **max_steps**: int - Maximum allowed steps

### TableSnapshot
- **table_name**: str - Name of the table
- **row_count**: int - Number of rows
- **columns**: List[ColumnInfo] - Column metadata
- **null_percentage**: float - Percentage of null values
- **duplicate_rows**: int - Number of duplicate rows

### ColumnInfo
- **name**: str - Column name
- **dtype**: str - Data type
- **null_count**: int - Number of null values
- **unique_count**: int - Number of unique values
- **sample_values**: List[Any] - Sample values

### Observation (Full)
- **state**: DataState - Current data state
- **current_step**: int - Current step number
- **total_steps**: int - Total steps allowed
- **action_history**: List[str] - List of actions taken
- **reward_accumulated**: float - Total accumulated reward
- **current_reward**: float - Last step reward

## Reward Logic

| Reward | Condition |
|--------|-----------|
| +1.0 | Full task success (score >= 1.0) |
| +0.5 | Substantial completion (score >= 0.8) |
| +0.2 | Correct intermediate steps |
| -0.3 | Unknown or failed action |
| -0.5 | Destructive actions or max steps reached |

## Task Difficulties

### Easy: Missing Value Handling
- Clean missing values from a single CSV
- Remove duplicates
- Max steps: 20

### Medium: Format Normalization
- Normalize date formats across multiple files
- Normalize currency values
- Merge tables
- Max steps: 30

### Hard: Statistical Outlier Detection & Merging
- Detect and remove outliers using Z-score
- Merge mismatched datasets
- Clean nulls and validate
- Max steps: 50

## Agent Grading

The environment includes automated grading:

```python
from tasks import TaskManager, AgentGrader

task_manager = TaskManager("easy")
grader = AgentGrader(task_manager)

# Grade after episode
grade_result = grader.grade_action_sequence(actions, final_data)
print(grade_result)
# {'coverage': 0.8, 'solution_score': 0.85, 'grade': 'A', ...}
```

## Baseline Agents

### HeuristicAgent
Rule-based agent that executes a fixed sequence of actions.

### OpenAIAgent
Uses OpenAI GPT models to select actions based on observation.

### LlamaAgent
Uses Llama models via API to select actions.

## Docker Deployment

```bash
# Build image
docker build -t dataclean-ops .

# Run container
docker run -p 7860:7860 dataclean-ops
```

## Environment Validation

```python
from env import create_env
import json

env = create_env("easy")
validation = env.validate()
print(json.dumps(validation, indent=2))
```

## Project Structure

```
├── openenv.yaml       # Meta-spec metadata
├── models.py          # Pydantic typed models
├── env.py             # OpenEnv interface implementation
├── tasks.py           # Task definitions and graders
├── baseline.py        # Agent implementations
├── Dockerfile         # Multi-stage build for HuggingFace Spaces
├── requirements.txt   # Python dependencies
└── README.md          # Documentation
```

## License

MIT