from pathlib import Path


def resolve_project_root() -> Path:
    root = Path.cwd()
    if (root / "data").exists() and (root / "results").exists():
        return root
    return root.parents[2]


PROJECT_ROOT = resolve_project_root()
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "result_03_Topic_and_Insight"


PARAMETERS = {
    "count_vectorizer": {
        "max_df": 0.8,
        "min_df": 2,
        "max_features": 1200,
        "stop_words": "english",
    },
    "lda": {
        "n_topics": 8,
        "n_top_words": 8,
        "random_state": 42,
        "learning_method": "online",
        "max_iter": 10,
    },
    "tfidf_vectorizer": {
        "max_df": 0.8,
        "min_df": 2,
        "max_features": 1200,
        "stop_words": "english",
    },
    "nmf": {
        "n_components": 4,
        "n_top_words": 8,
        "random_state": 42,
        "max_iter": 500,
    },
}
