import argparse
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", default="mlarr_text", help="Path to the train dataset")
    args = parser.parse_args()

    # Load data
    data = load_files(args.data, encoding="latin-1")

    # Define pipelines
    pipelines = [
        {
            'name': 'Count Vectorizer -> Naive Bayes',
            'vectorizer': CountVectorizer(),
            'estimator': MultinomialNB(),
            'params': {
                'Count Vectorizer -> Naive Bayes__ngram_range': [(1, 1)],
                'Count Vectorizer -> Naive Bayes__stop_words': ['english'],
                'Classifier__alpha': [0.1]
            }
        },
        {
            'name': 'TF-IDF -> Naive Bayes',
            'vectorizer': TfidfVectorizer(),
            'estimator': MultinomialNB(),
            'params': {
                'TF-IDF -> Naive Bayes__ngram_range': [(1, 1)],
                'TF-IDF -> Naive Bayes__stop_words': ['english'],
                'Classifier__alpha': [0.1]
            }
        },
        {
            'name': 'TF-IDF -> SVM',
            'vectorizer': TfidfVectorizer(),
            'estimator': SVC(),
            'params': {
                'TF-IDF -> SVM__ngram_range': [(1, 2)],
                'TF-IDF -> SVM__stop_words': ['english'],
                'Classifier__C': [25],
                'Classifier__kernel': ['sigmoid'],
                'Classifier__gamma': ['scale']
            }
        }
    ]

    # Fit models and save them
    for pipeline in pipelines:
        model = Pipeline([
            (pipeline['name'], pipeline['vectorizer']),
            ('Classifier', pipeline['estimator'])
        ])

        # Perform grid search
        grid_search = GridSearchCV(model, param_grid=pipeline['params'], cv=10, n_jobs=-1)
        grid_search.fit(data.data, data.target)

        # Print best parameters and score
        print(f"\nBest parameters for {pipeline['name']}:\n{grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

        # Save the model
        joblib.dump(grid_search.best_estimator_, f'trained_model_{pipeline["name"].replace(" -> ", "_").replace(" ", "_")}.pkl')

    print("\nModels saved successfully.")
