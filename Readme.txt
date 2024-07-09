# Text Classification
Dependencies
	scikit-learn
	pandas
	
	pip install scikit-learn pandas
File Descriptions

Datasets
test_dataset
bbc

Python script files

modelSelection.py

	Performs grid search to find the best hyperparameters for various classifiers using Count Vectorizer and TF-IDF Vectorizer.
	Outputs the best parameters, cross-validation scores, and grid search durations for each pipeline.
	data provide is used to train models "mlarr_text" data file name
trainSelectedModels.py

	Trains the best models identified from the grid search on the entire dataset.
	Saves the trained models for later evaluation.
	data provide is used to train the selected model "mlarr_text" data file name
generateTestData.py

	Generates unique test datasets for evaluating the trained models.
	Ensures the test data is distinct from the training data.
	downloaded "bbc" data and "mlarr_text" file name should be on the same directory and the matched data will be removed form the bbs
	folde.
	
testTrainedModels.py

	Evaluates the trained models using the generated test datasets.
	Outputs performance metrics such as accuracy, precision, recall, and F1-score for each model.
	the remaing data set from the "bbc" in the testData generation is attached with the scrip files as "test_dataset" file name
	
bash/script to run the python files	
python modelSelection.py
python trainSelectedModels.py
python generateTestData.py
python testTrainedModels.py

the data can be found in http://mlg.ucd.ie/datasets/bbc.html and should be saved as bbc with each class.
