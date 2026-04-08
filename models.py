from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    CLEAN_NULLS = "clean_nulls"
    FORMAT_DATE = "format_date"
    DROP_DUPLICATES = "drop_duplicates"
    MERGE_TABLES = "merge_tables"
    REMOVE_OUTLIERS = "remove_outliers"
    NORMALIZE_CURRENCY = "normalize_currency"
    VALIDATE = "validate"


class ColumnStats(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class TableInfo(BaseModel):
    name: str
    rows: int
    cols: int
    null_pct: float
    dupes: int
    column_stats: List[ColumnStats] = Field(default_factory=list)


class Observation(BaseModel):
    task: str
    step: int
    max_steps: int
    tables: Dict[str, TableInfo]
    active_table: str
    action_history: List[str] = Field(default_factory=list)
    reward_accumulated: float = 0.0


class ActionParams(BaseModel):
    columns: Optional[List[str]] = None
    strategy: Optional[str] = "default"
    threshold: Optional[float] = 3.0
    fill_value: Optional[Any] = None
    merge_key: Optional[str] = None
    date_format: Optional[str] = None


class Action(BaseModel):
    action_type: ActionType
    params: ActionParams = Field(default_factory=ActionParams)
    
    @property
    def type(self):
        return self.action_type


class Reward(BaseModel):
    value: float
    reason: str
    is_terminal: bool = False
    solved: bool = False


class StepOutput(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)