"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""

    data_dir: str = "./machine_learning_engineering/tasks/"  # the directory path where the machine learning tasks and their data are stored.
    task_name: str = "spaceship-titanic"  # The name of the specific task to be loaded and processed.
    task_type: str = (
        "Tabular Classification"  # The type of machine learning problem.
    )
    lower: bool = False  # True if a lower value of the metric is better.
    workspace_dir: str = "./machine_learning_engineering/workspace/"  # Directory used for saving intermediate outputs, results, logs.
    agent_model: str = os.environ.get(
        "ROOT_AGENT_MODEL", "gemini-2.0-flash-001"
    )  # Name the LLM model to be used by the agent.
    task_description: str = ""  # The detailed description of the task.
    task_summary: str = ""  # The concise summary of the task.
    start_time: float = 0.0  # Timestamp indicating the start time of the task. Typically represented in seconds since the epoch.
    seed: int = 42  # The random seed value used to ensure reproducibility of experiments.
    exec_timeout: int = (
        600  # The maximum time in seconds allowed to complete the task. (DEFAULT: 600)
    )
    num_solutions: int = 2  # The number of different solutions to generate or attempt for the given task. (DEFAULT: 2)
    num_model_candidates: int = 2  # The number of different model architectures or hyperparameter sets to consider as candidates. (DEFAULT: 2)
    max_retry: int = (
        10  # The maximum number of times to retry a failed operation. (DEFAULT: 10)
    )
    max_debug_round: int = 5  # The maximum number of iterations or rounds allowed for the debugging step. (DEFAULT: 5)
    max_rollback_round: int = 2  # The maximum number of times the system can rollback to a previous state, in case of errors or poor performance. (DEFAULT: 2)
    inner_loop_round: int = 1  # The number of iterations or rounds to be executed within an inner loop of the system. (DEFAULT: 1)
    outer_loop_round: int = 1  # The number of iterations or rounds to be executed within the outer loop, which might encompass multiple inner loops. (DEFAULT: 1)
    ensemble_loop_round: int = 1  # The number of rounds or iterations dedicated to ensembling, combining multiple models or solutions. (DEFAULT: 1)
    num_top_plans: int = 2  # The number of highest-scoring plans or strategies to select or retain. (DEFAULT: 2)
    use_data_leakage_checker: bool = True  # Enable (`True`) or disable (`False`) a check for data leakage in the machine learning pipeline. (DEFAULT: False)
    use_data_usage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for how data is being used, potentially for compliance or best practices. (DEFAULT: False)
    table_report_enabled: bool = True  # Build TableReport data profile for refinement prompts (ablation + planners).
    tuning_enabled: bool = True  # Run terminal choose_* search after refinement.
    tuning_n_iter: int = 5  # Max randomized search iterations for terminal tuning.


CONFIG = DefaultConfig()


# --- Helper functions for model compatibility (GPT-5 family + litellm expect temp=1.0) ---
def is_gpt5_family_model(model_name: str) -> bool:
    """Returns True for OpenAI GPT-5 family models only."""
    if not model_name:
        return False
    normalized = model_name.strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    return normalized.startswith("gpt-5")


def get_compatible_temperature(model_name: str, requested_temp: float) -> float:
    """
    Normalizes temperature only for model families with strict requirements.
    GPT-5 family models require temperature=1.0 to fix litellm error, other models use the requested temperature.
    """
    if is_gpt5_family_model(model_name):
        return 1.0
    return requested_temp
