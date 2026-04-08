from enum import Enum
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from dataclasses import dataclass


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TaskDefinition:
    name: str
    description: str
    difficulty: TaskDifficulty
    max_steps: int
    expected_actions: List[str]
    grading_criteria: Dict[str, Any]


class TaskManager:
    """Manages data cleaning tasks with programmatic graders."""
    
    def __init__(self, difficulty: TaskDifficulty):
        self.difficulty = difficulty
        self.task_def = self._get_definition()
    
    def _get_definition(self) -> TaskDefinition:
        tasks = {
            TaskDifficulty.EASY: TaskDefinition(
                name="Missing Value Handling",
                description="Handle missing values in a single CSV dataset",
                difficulty=TaskDifficulty.EASY,
                max_steps=20,
                expected_actions=["clean_nulls", "validate"],
                grading_criteria={"null_threshold": 5.0}
            ),
            TaskDifficulty.MEDIUM: TaskDefinition(
                name="Date/Currency Normalization",
                description="Normalize date and currency formats across 2 messy files",
                difficulty=TaskDifficulty.MEDIUM,
                max_steps=30,
                expected_actions=["format_date", "normalize_currency", "merge_tables", "validate"],
                grading_criteria={"date_format": "%Y-%m-%d", "currency_precision": 2}
            ),
            TaskDifficulty.HARD: TaskDefinition(
                name="Outlier Detection & Dataset Merging",
                description="Detect and remove statistical outliers and merge mismatched datasets",
                difficulty=TaskDifficulty.HARD,
                max_steps=35,
                expected_actions=["remove_outliers", "merge_tables", "clean_nulls", "validate"],
                grading_criteria={"outlier_threshold": 3.0, "merge_key_required": True}
            )
        }
        return tasks[self.difficulty]
    
    def generate_data(self, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for the task."""
        if self.difficulty == TaskDifficulty.EASY:
            return self._generate_easy(rng)
        elif self.difficulty == TaskDifficulty.MEDIUM:
            return self._generate_medium(rng)
        else:
            return self._generate_hard(rng)
    
    def _generate_easy(self, rng) -> Dict[str, pd.DataFrame]:
        """Generate data with missing values."""
        n = rng.integers(50, 150)
        
        data = {
            "id": range(1, n + 1),
            "name": [f"User_{i}" for i in range(1, n + 1)],
            "email": [f"user{i}@example.com" for i in range(1, n + 1)],
            "age": rng.integers(18, 70, size=n).astype(float),
            "salary": rng.integers(30000, 150000, size=n).astype(float),
            "department": rng.choice(["IT", "HR", "Sales", "Marketing", "Finance"], size=n)
        }
        
        df = pd.DataFrame(data)
        
        null_idx = rng.choice(n, size=int(n * 0.15), replace=False)
        df.loc[null_idx, "age"] = np.nan
        
        null_idx = rng.choice(n, size=int(n * 0.10), replace=False)
        df.loc[null_idx, "salary"] = np.nan
        
        null_idx = rng.choice(n, size=int(n * 0.08), replace=False)
        df.loc[null_idx, "department"] = np.nan
        
        return {"employees": df}
    
    def _generate_medium(self, rng) -> Dict[str, pd.DataFrame]:
        """Generate data with messy date/currency formats."""
        n = rng.integers(40, 100)
        
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        date_formats = []
        for d in dates:
            if rng.random() < 0.4:
                date_formats.append(d.strftime("%m/%d/%Y"))
            elif rng.random() < 0.7:
                date_formats.append(d.strftime("%d-%m-%Y"))
            else:
                date_formats.append(d.strftime("%Y-%m-%d"))
        
        data1 = {
            "id": range(1, n + 1),
            "join_date": date_formats,
            "salary": [f"${rng.integers(30000, 150000)}" for _ in range(n)],
            "department": rng.choice(["IT", "HR", "Sales"], size=n)
        }
        
        n2 = rng.integers(30, 80)
        dates2 = pd.date_range("2023-06-01", periods=n2, freq="D")
        date_formats2 = []
        for d in dates2:
            if rng.random() < 0.5:
                date_formats2.append(d.strftime("%Y/%m/%d"))
            else:
                date_formats2.append(d.strftime("%d.%m.%Y"))
        
        data2 = {
            "id": range(1, n2 + 1),
            "start_date": date_formats2,
            "bonus": [f"{rng.integers(1000, 20000)}.{rng.integers(0, 99):02d}" for _ in range(n2)],
            "level": rng.choice(["Junior", "Mid", "Senior"], size=n2)
        }
        
        return {
            "employees": pd.DataFrame(data1),
            "compensation": pd.DataFrame(data2)
        }
    
    def _generate_hard(self, rng) -> Dict[str, pd.DataFrame]:
        """Generate data with outliers and mismatched datasets."""
        n = rng.integers(80, 120)
        
        base_ages = rng.normal(35, 10, size=n).clip(18, 70)
        outlier_idx = rng.choice(n, size=5, replace=False)
        base_ages[outlier_idx] += rng.uniform(40, 80, size=5)
        
        base_salaries = rng.lognormal(10.5, 0.5, size=n).clip(20000, 200000)
        outlier_idx = rng.choice(n, size=5, replace=False)
        base_salaries[outlier_idx] *= rng.uniform(3, 5, size=5)
        
        data1 = {
            "id": range(1, n + 1),
            "name": [f"Employee_{i}" for i in range(1, n + 1)],
            "age": base_ages,
            "salary": base_salaries,
            "join_date": rng.choice(["2020-01-15", "2021-06-20", "2022-03-10", "2023-09-01"], size=n)
        }
        
        n2 = rng.integers(60, 100)
        overlap = min(n, n2)
        
        data2 = {
            "emp_id": list(range(1, overlap + 1)) + list(range(overlap + 1, n2 + 1)),
            "department": rng.choice(["Engineering", "Sales", "Marketing", "Support"], size=n2),
            "performance": rng.uniform(1, 5, size=n2),
            "tenure_months": rng.integers(1, 60, size=n2)
        }
        
        null_idx = rng.choice(n2, size=int(n2 * 0.1), replace=False)
        data2["performance"][null_idx] = np.nan
        
        return {
            "personnel": pd.DataFrame(data1),
            "hr_records": pd.DataFrame(data2)
        }
    
    def grade_solution(self, data: Dict[str, pd.DataFrame]) -> float:
        """Grade solution from 0.0 to 1.0."""
        if self.difficulty == TaskDifficulty.EASY:
            return self._grade_easy(data)
        elif self.difficulty == TaskDifficulty.MEDIUM:
            return self._grade_medium(data)
        else:
            return self._grade_hard(data)
    
    def _grade_easy(self, data: Dict[str, pd.DataFrame]) -> float:
        """Grade missing value task."""
        if "employees" not in data:
            return 0.0
        
        df = data["employees"]
        score = 0.0
        
        null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if null_pct < 5.0:
            score += 0.5
        elif null_pct < 10.0:
            score += 0.25
        
        if len(df) >= 40:
            score += 0.25
        
        required = ["id", "name", "email", "age", "salary"]
        if all(c in df.columns for c in required):
            score += 0.25
        
        return min(score, 1.0)
    
    def _grade_medium(self, data: Dict[str, pd.DataFrame]) -> float:
        """Grade date/currency normalization task."""
        score = 0.0
        
        if "employees" in data:
            df = data["employees"]
            
            if "join_date" in df.columns:
                try:
                    parsed = pd.to_datetime(df["join_date"], errors='coerce')
                    valid_pct = parsed.notna().sum() / len(df)
                    if valid_pct > 0.8:
                        score += 0.25
                    elif valid_pct > 0.5:
                        score += 0.15
                except:
                    pass
            
            if "salary" in df.columns:
                try:
                    salaries = df["salary"].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    salaries = pd.to_numeric(salaries, errors='coerce')
                    valid_pct = salaries.notna().sum() / len(df)
                    if valid_pct > 0.8:
                        score += 0.25
                    elif valid_pct > 0.5:
                        score += 0.15
                except:
                    pass
        
        if "compensation" in data:
            df = data["compensation"]
            if "start_date" in df.columns:
                try:
                    parsed = pd.to_datetime(df["start_date"], errors='coerce')
                    valid_pct = parsed.notna().sum() / len(df)
                    if valid_pct > 0.8:
                        score += 0.2
                except:
                    pass
        
        if "merged" in data:
            merged = data["merged"]
            if len(merged) > 0:
                score += 0.3
        
        return min(score, 1.0)
    
    def _grade_hard(self, data: Dict[str, pd.DataFrame]) -> float:
        """Grade outlier detection and merging task."""
        score = 0.0
        
        if "personnel" in data:
            df = data["personnel"]
            
            if "age" in df.columns:
                mean, std = df["age"].mean(), df["age"].std()
                if std > 0:
                    z = np.abs((df["age"] - mean) / std)
                    outliers = (z > 3).sum()
                    if outliers < 3:
                        score += 0.2
            
            if "salary" in df.columns:
                mean, std = df["salary"].mean(), df["salary"].std()
                if std > 0:
                    z = np.abs((df["salary"] - mean) / std)
                    outliers = (z > 3).sum()
                    if outliers < 3:
                        score += 0.2
        
        if "hr_records" in data:
            df = data["hr_records"]
            if "performance" in df.columns:
                null_pct = df["performance"].isnull().sum() / len(df)
                if null_pct < 0.05:
                    score += 0.2
        
        if "merged" in data or ("personnel" in data and "hr_records" in data):
            try:
                if "merged" in data:
                    merged = data["merged"]
                else:
                    merged = pd.merge(data["personnel"], data["hr_records"],
                                      left_on="id", right_on="emp_id", how="inner")
                if len(merged) > 0:
                    score += 0.4
            except:
                pass
        
        return min(score, 1.0)


class AgentGrader:
    """Grader for agent action sequences."""
    
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
    
    def grade(self, actions: List[str], final_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Grade agent performance."""
        expected = self.task_manager.task_def.expected_actions
        performed = set(actions)
        expected_set = set(expected)
        
        coverage = len(performed.intersection(expected_set)) / len(expected_set)
        solution_score = self.task_manager.grade_solution(final_data)
        
        if solution_score >= 0.8:
            grade = "A"
        elif solution_score >= 0.6:
            grade = "B"
        elif solution_score >= 0.4:
            grade = "C"
        else:
            grade = "F"
        
        return {
            "coverage": coverage,
            "solution_score": solution_score,
            "grade": grade,
            "actions_performed": list(performed),
            "expected": expected
        }