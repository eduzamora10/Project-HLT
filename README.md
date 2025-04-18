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

    Here we are choosing the 20 most difficult languages each model faced by Logistic Regresison, SVM, and Naive Bayes. 

    This will be needed to create our hard test set. 

    To run:
    python generate_hard_languages.py

**generate_hard_test_set.py**

    In this script, we are combining all hardest languages from all models into one test set.

    To run:
    python generate_hard_test_set.py

**evaluate_hard_dataset.py**

    Here we do the exact same thing we did in program.py, just evaluating the classic models with the hard test set we created.

    To run Logistic Regression model:
    python evaluate_hard_dataset.py logistic

    To run SVM model:
    python evaluate_hard_dataset.py svm

    To run Naive Bayes model:
    python evaluate_hard_dataset.py nb
