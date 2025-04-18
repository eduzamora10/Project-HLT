# Project-HLT
Code for Project (CS 4395)

**program.py**

    This script handles preprocessing, training, and evaluation of the following models:

        * Logistic Regression

        * Support Vector Machine (SVM)

        * Naive Bayes

    It outputs evaluation results to the following files:

        * logreg_results.txt

        * svm_results.txt

        * nb_results.txt

    After training, it saves the models along with the associated vectorizer and label encoder, which are later used in evaluate_hard_dataset.py.

    To run:
        python program.py logistic   # Run Logistic Regression
        python program.py svm        # Run SVM
        python program.py nb         # Run Naive Bayes


**generate_hard_langauges.py**

    This script identifies the 20 most difficult languages for each model (Logistic Regression, SVM, and Naive Bayes) based on performance.

    These languages are used to help define the "hard test set." 

    To run:
        python generate_hard_languages.py

**generate_hard_test_set.py**

    This script combines the hardest languages (identified from each model) into one unified test set.

    To run:
        python generate_hard_test_set.py

**evaluate_hard_dataset.py**

    This script evaluates the previously trained classic models (Logistic Regression, SVM, Naive Bayes) on the newly created hard test set.

    It reuses the saved models, vectorizer, and label encoder from program.py.

    To run:
        python evaluate_hard_dataset.py logistic   # Evaluate Logistic Regression
        python evaluate_hard_dataset.py svm        # Evaluate SVM
        python evaluate_hard_dataset.py nb         # Evaluate Naive Bayes

