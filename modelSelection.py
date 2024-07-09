from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from timeit import default_timer as timer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Function to create grid search pipeline
def create_grid_search(estimator, param_grid, vectorizer_name):
    return GridSearchCV(
        estimator=Pipeline([
            (vectorizer_name, CountVectorizer() if vectorizer_name == 'Count_Vectorizer' else TfidfVectorizer()),
            ('Classifier', estimator)
        ]),
        param_grid=param_grid,
        cv=10, n_jobs=-1
    )

# Define all parameter grids
param_grids = {
    'Naive_Bayes_Count_Vectorizer': {
        'Count_Vectorizer__stop_words': ['english', None],
        'Count_Vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__alpha': [0.1, 0.4, 0.7, 1.0]
    },
    'Naive_Bayes_TF_IDF': {
        'TF-IDF__stop_words': ['english', None],
        'TF-IDF__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__alpha': [0.1, 0.4, 0.7, 1.0]
    },
    'Decision_Tree_Count_Vectorizer': {
        'Count_Vectorizer__stop_words': ['english', None],
        'Count_Vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__max_depth': [None, 5, 10]
    },
    'Decision_Tree_TF_IDF': {
        'TF-IDF__stop_words': ['english', None],
        'TF-IDF__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__max_depth': [None, 5, 10]
    },
    'KNN_Count_Vectorizer': {
        'Count_Vectorizer__stop_words': ['english', None],
        'Count_Vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__n_neighbors': [3, 5, 7]
    },
    'KNN_TF_IDF': {
        'TF-IDF__stop_words': ['english', None],
        'TF-IDF__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__n_neighbors': [3, 5, 7]
    },
    'SVM_Count_Vectorizer': {
        'Count_Vectorizer__stop_words': ['english', None],
        'Count_Vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__C': [1, 25, 50, 100],
        'Classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'Classifier__gamma': ['scale', 'auto']
    },
    'SVM_TF_IDF': {
        'TF-IDF__stop_words': ['english', None],
        'TF-IDF__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'Classifier__C': [1, 25, 50, 100],
        'Classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'Classifier__gamma': ['scale', 'auto']
    }
}

# Classifiers used
classifiers = {
    'Naive_Bayes': MultinomialNB(),
    'Decision_Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

if __name__ == "__main__":
    start = timer()
    data = load_files("mlarr_text", encoding="latin-1")
    best_params_scores = []

    # Perform grid searches
    for classifier_name, classifier in classifiers.items():
        for vectorizer_name in ['Count_Vectorizer', 'TF-IDF']:
            key = f"{classifier_name}_{vectorizer_name.replace('-', '_')}"
            grid_search = create_grid_search(classifier, param_grids[key], vectorizer_name)
            pipeline_name = f"{vectorizer_name.replace('_', ' ')} -> {classifier_name.replace('_', ' ')}"
            print(f"Starting grid search for {pipeline_name}...")
            grid_search_start = timer()
            grid_search.fit(data.data, data.target)
            grid_search_end = timer()
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            best_params_scores.append((pipeline_name, best_params, best_score, round(grid_search_end - grid_search_start, 2)))
            print(f"Completed grid search for {pipeline_name} in {round(grid_search_end - grid_search_start, 2)}s")

    end = timer()
    print("\nBest parameters and scores for each pipeline:")
    report = []
    for pipeline_name, best_params, best_score, duration in best_params_scores:
        report.append({
            'Pipeline': pipeline_name,
            'Best Parameters': best_params,
            'Best Score': best_score,
            'Duration (s)': duration
        })
        print(f"\nPipeline: {pipeline_name}")
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score:.4f}")
        print(f"Duration: {duration}s")

    # Convert report to DataFrame for better visualization
    report_df = pd.DataFrame(report)
    print("\nSummary Report:")
    print(report_df)

    print(f"\nTotal Execution Time: {round(end - start, 2)}s")
