import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from models import (
    Action, ActionType, ActionParams, Observation, Reward, StepOutput,
    TableInfo, ColumnStats
)
from tasks import TaskManager, TaskDifficulty


console = Console()


class DataCleanOpsEnv:
    """
    OpenEnv-compatible environment for data cleaning tasks.
    Provides standard step()/reset()/state() API for RL agents.
    """
    
    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        self.task = task
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.task_manager = TaskManager(TaskDifficulty(task))
        self.data: Dict[str, pd.DataFrame] = {}
        self.step_count = 0
        self.max_steps = 20
        self.history: List[str] = []
        self.total_reward = 0.0
        self.done = False
        
        self._init_data()
    
    def _init_data(self):
        self.data = self.task_manager.generate_data(self.rng)
        self.max_steps = self.task_manager.task_def.max_steps
    
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.step_count = 0
        self.history = []
        self.total_reward = 0.0
        self.done = False
        self._init_data()
        
        return self._get_observation()
    
    def step(self, action: Action) -> StepOutput:
        """Execute one step in the environment."""
        if self.done:
            return StepOutput(
                observation=self._get_observation(),
                reward=Reward(value=0.0, reason="Episode terminated", is_terminal=True),
                done=True,
                info={}
            )
        
        self.step_count += 1
        self.history.append(action.action_type.value)
        
        reward, done, solved, info = self._execute_action(action)
        self.total_reward += reward
        self.done = done
        
        if self.step_count >= self.max_steps and not done:
            reward = -0.5
            done = True
            reason = "Max steps reached - penalty applied"
        else:
            reason = info.get("reason", f"Action {action.action_type.value} executed")
        
        obs = self._get_observation()
        
        return StepOutput(
            observation=obs,
            reward=Reward(
                value=reward,
                reason=reason,
                is_terminal=done,
                solved=solved
            ),
            done=done,
            info=info
        )
    
    def _execute_action(self, action: Action) -> tuple:
        """Execute action and return (reward, done, solved, info)."""
        info = {"action": action.action_type.value}
        
        try:
            if action.action_type == ActionType.CLEAN_NULLS:
                return self._clean_nulls(action.params)
            elif action.action_type == ActionType.FORMAT_DATE:
                return self._format_date(action.params)
            elif action.action_type == ActionType.DROP_DUPLICATES:
                return self._drop_duplicates(action.params)
            elif action.action_type == ActionType.MERGE_TABLES:
                return self._merge_tables(action.params)
            elif action.action_type == ActionType.REMOVE_OUTLIERS:
                return self._remove_outliers(action.params)
            elif action.action_type == ActionType.NORMALIZE_CURRENCY:
                return self._normalize_currency(action.params)
            elif action.action_type == ActionType.VALIDATE:
                return self._validate(action.params)
            else:
                return -0.3, False, False, {"reason": "Unknown action"}
        except Exception as e:
            return -0.5, False, False, {"reason": f"Error: {str(e)}"}
    
    def _clean_nulls(self, params: ActionParams) -> tuple:
        """Handle missing values."""
        table = list(self.data.keys())[0]
        df = self.data[table]
        
        initial_nulls = df.isnull().sum().sum()
        cols = params.columns or df.columns[df.isnull().any()].tolist()
        
        if not cols:
            return 0.0, False, False, {"reason": "No nulls to clean"}
        
        strategy = params.strategy or "drop"
        original_rows = len(df)
        
        if strategy == "drop":
            df = df.dropna(subset=cols)
        else:
            for col in cols:
                if col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna("unknown")
        
        self.data[table] = df
        nulls_removed = initial_nulls - df.isnull().sum().sum()
        
        if nulls_removed > 0 and (original_rows - len(df)) < original_rows * 0.3:
            return 0.2, False, False, {"reason": f"Cleaned {nulls_removed} null values"}
        elif nulls_removed > 0:
            return -0.3, False, False, {"reason": "Too many rows removed - data loss"}
        return -0.1, False, False, {"reason": "No nulls found"}
    
    def _format_date(self, params: ActionParams) -> tuple:
        """Format date columns."""
        table = list(self.data.keys())[0]
        df = self.data[table]
        
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if not date_cols:
            date_cols = params.columns or []
        
        if not date_cols:
            return -0.2, False, False, {"reason": "No date columns found"}
        
        formatted = 0
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    formatted += df[col].notna().sum()
                except:
                    pass
        
        self.data[table] = df
        if formatted > 0:
            return 0.2, False, False, {"reason": f"Formatted {formatted} date values"}
        return -0.1, False, False, {"reason": "No dates formatted"}
    
    def _drop_duplicates(self, params: ActionParams) -> tuple:
        """Remove duplicate rows."""
        table = list(self.data.keys())[0]
        df = self.data[table]
        
        original = len(df)
        df = df.drop_duplicates()
        self.data[table] = df
        
        removed = original - len(df)
        if removed > 0:
            return 0.2, False, False, {"reason": f"Removed {removed} duplicate rows"}
        return 0.0, False, False, {"reason": "No duplicates found"}
    
    def _merge_tables(self, params: ActionParams) -> tuple:
        """Merge multiple tables."""
        tables = list(self.data.keys())
        
        if len(tables) < 2:
            return -0.3, False, False, {"reason": "Need at least 2 tables to merge"}
        
        merge_key = params.merge_key or "id"
        
        if merge_key not in self.data[tables[0]].columns:
            return -0.3, False, False, {"reason": f"Merge key '{merge_key}' not found"}
        
        result = self.data[tables[0]]
        for i in range(1, len(tables)):
            if merge_key in self.data[tables[i]].columns:
                result = pd.merge(result, self.data[tables[i]], on=merge_key, how='outer')
        
        self.data["merged"] = result
        return 0.2, False, False, {"reason": f"Merged {len(tables)} tables into 'merged'"}
    
    def _remove_outliers(self, params: ActionParams) -> tuple:
        """Detect and remove statistical outliers using Z-score."""
        table = list(self.data.keys())[0]
        df = self.data[table]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return -0.2, False, False, {"reason": "No numeric columns for outlier detection"}
        
        threshold = params.threshold or 3.0
        original = len(df)
        
        for col in numeric_cols:
            if col in df.columns:
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    z = np.abs((df[col] - mean) / std)
                    df = df[z < threshold]
        
        self.data[table] = df
        removed = original - len(df)
        
        if 0 < removed < original * 0.1:
            return 0.2, False, False, {"reason": f"Removed {removed} statistical outliers"}
        elif removed == 0:
            return 0.0, False, False, {"reason": "No outliers detected"}
        return -0.3, False, False, {"reason": "Too many rows removed - possible valid data loss"}
    
    def _normalize_currency(self, params: ActionParams) -> tuple:
        """Normalize currency values."""
        table = list(self.data.keys())[0]
        df = self.data[table]
        
        currency_cols = [c for c in df.columns if any(x in c.lower() for x in ['amount', 'price', 'salary', 'revenue', 'cost'])]
        
        if not currency_cols:
            return -0.2, False, False, {"reason": "No currency columns found"}
        
        normalized = 0
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                normalized += df[col].notna().sum()
        
        self.data[table] = df
        if normalized > 0:
            return 0.2, False, False, {"reason": f"Normalized {normalized} currency values"}
        return -0.1, False, False, {"reason": "No currency values normalized"}
    
    def _validate(self, params: ActionParams) -> tuple:
        """Validate and grade the solution."""
        score = self.task_manager.grade_solution(self.data)
        
        if score >= 1.0:
            return 1.0, True, True, {"reason": "Task completed successfully!", "score": score}
        elif score >= 0.8:
            return 0.5, True, False, {"reason": "Task substantially complete", "score": score}
        return 0.0, False, False, {"reason": f"Validation score: {score:.2f}", "score": score}
    
    def _get_observation(self) -> Observation:
        """Get current observation."""
        tables = {}
        
        for name, df in self.data.items():
            col_stats = []
            for col in df.columns:
                cs = ColumnStats(
                    name=col,
                    dtype=str(df[col].dtype),
                    null_count=int(df[col].isnull().sum()),
                    unique_count=int(df[col].nunique()),
                    sample_values=df[col].dropna().head(3).tolist()
                )
                if df[col].dtype in ['int64', 'float64']:
                    cs.mean = float(df[col].mean())
                    cs.std = float(df[col].std()) if df[col].std() > 0 else None
                    cs.min_val = float(df[col].min())
                    cs.max_val = float(df[col].max())
                col_stats.append(cs)
            
            tables[name] = TableInfo(
                name=name,
                rows=len(df),
                cols=len(df.columns),
                null_pct=float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] > 0 else 0,
                dupes=int(df.duplicated().sum()),
                column_stats=col_stats
            )
        
        active = list(self.data.keys())[0] if self.data else ""
        
        return Observation(
            task=self.task,
            step=self.step_count,
            max_steps=self.max_steps,
            tables=tables,
            active_table=active,
            action_history=self.history.copy(),
            reward_accumulated=self.total_reward
        )
    
    def state(self) -> Observation:
        """Get current observation - OpenEnv required method."""
        return self._get_observation()
    
    def validate(self) -> Dict[str, Any]:
        """Validate environment configuration."""
        return {
            "valid": True,
            "framework": "openenv",
            "version": "1.0.0",
            "actions": [a.value for a in ActionType],
            "task": self.task,
            "tasks_available": ["easy", "medium", "hard"]
        }
    
    def close(self):
        """Close the environment and cleanup resources."""
        self.data = {}
        self.done = True
    
    def render(self):
        """Render environment to console with rich UI."""
        obs = self.state()
        
        console.print(Panel.fit(
            f"[bold cyan]DataClean-Ops[/bold cyan] | Task: {obs.task.upper()} | Step: {obs.step}/{obs.max_steps} | Reward: {obs.reward_accumulated:.2f}",
            border_style="cyan"
        ))
        
        for name, table in obs.tables.items():
            t = Table(title=f"📊 Table: {name}", show_header=True, header_style="bold cyan")
            t.add_column("Column", style="cyan")
            t.add_column("Type", style="yellow")
            t.add_column("Nulls", style="red")
            t.add_column("Unique", style="green")
            
            for col in table.column_stats[:6]:
                t.add_row(col.name, col.dtype, str(col.null_count), str(col.unique_count))
            
            console.print(t)
            console.print(f"  Rows: {table.rows} | Null %: {table.null_pct:.1f} | Duplicates: {table.dupes}")


def create_env(task: str = "easy", seed: Optional[int] = None) -> DataCleanOpsEnv:
    """Factory function to create environment."""
    return DataCleanOpsEnv(task=task, seed=seed)


if __name__ == "__main__":
    env = create_env("easy")
    obs = env.reset()
    print(f"[OK] Environment ready: {obs.task}")
    print(f"[INFO] Tables: {list(obs.tables.keys())}")
    print(f"[INFO] Actions: {env.validate()['actions']}")