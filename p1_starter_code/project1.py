"""EECS 445 - Fall 2023.

Project 1
"""

import itertools
import string
import warnings

import matplotlib
import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from numpy import dot
from numpy.linalg import norm
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import pos_tag, word_tokenize
import matplotlib.pyplot as plt








warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)


def extract_word(input_string):
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    returnVec = []

    for ch in string.punctuation:
        input_string = input_string.replace(ch, ' ')
    for word in input_string.split():
        returnVec.append(word.lower())
    return returnVec

def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    index = 0
    # TODO: Implement this function
    for iter, row in df.iterrows():
        sentence = row['reviewText']
        words = extract_word(sentence)

        for word in words:
            if word not in word_dict:
                word_dict[word] = index
                index += 1
    return word_dict

def generate_feature_multi(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=extract_word, min_df = 3, max_df = 0.7)
    sentences = df['reviewText'].tolist()  # Extract the sentences from the specified column
    vectorizer.fit(sentences)
    print('voc')

    words = vectorizer.transform(sentences)
    feature_matrix = words.toarray()
    return feature_matrix, vectorizer

    # TODO: Implement this function
    #for iter, row in df.iterrows():
    #    sentence = row['reviewText']
    #    words = vectorizer.fit_transform(sentence)
    #    print('wor')
    ###   print('vec')
      #  print(vectorizer.get_feature_names())
      #  for word in words:
      #      if word not in word_dict:
      #          word_dict[word] = index
      #          index += 1

def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    for iter, row in df.iterrows():
        sentence = row['reviewText']
        words = extract_word(sentence)
        for word in words:
            if word in word_dict:
                feature_matrix[iter][word_dict[word]] = 1
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    if(metric == 'accuracy'):
        return accuracy_score(y_true, y_pred)
    elif(metric == 'f1-score'):
        return f1_score(y_true, y_pred)
    elif(metric == 'precision'):
        return precision_score(y_true, y_pred)
    elif(metric == 'sensitivity'):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp/(tp + fn)
        return sensitivity
    elif(metric == 'specificity'):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()            
        specificity = tn / (tn+fp)
        return specificity
    elif(metric == 'auroc'):
        return (roc_auc_score(y_true, y_pred))
    else:
        return 0
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    scores = []

    skf = StratifiedKFold(n_splits=k)
    for trainIndex, testIndex in skf.split(X, y):
        xTrainFold, xTestFold = X[trainIndex], X[testIndex]
        yTrainFold, yTestFold = y[trainIndex], y[testIndex]
        clf.fit(xTrainFold, yTrainFold)
        classifier_predictions = None
        if(metric == 'auroc'):
            classifier_predictions = clf.decision_function(xTestFold)
        else:    
            classifier_predictions = clf.predict(xTestFold)
        
        scores.append(performance(yTestFold, classifier_predictions, metric=metric))

    # Put the performance of the model on each fold in the scores array
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """

    res = np.zeros(len(C_range))
    for i, c in enumerate(C_range):
        clf = svm.LinearSVC(penalty = penalty, dual=dual, loss=loss, random_state = 445, C = c)
        res[i] = (cv_performance(clf, X=X, y=y, k=k, metric=metric))
    print(res)
    print(res[res.argmax()])
    print(C_range[res.argmax()])
    return C_range[res.argmax()]


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []

    for c in C_range:
        clf = svm.LinearSVC(penalty = penalty, dual=dual, loss=loss, random_state = 445, C = c)
        clf.fit(X=X, y=y)
        count = 0
        for coef in clf.coef_[0]:
            if(coef == 0):
                count += 1
        norm0.append(count)
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[], search='grid'):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    best_performance = 0

    if(search =='grid'):
        res = np.zeros(len(param_range.T[0]) * len(param_range.T[0]))

        for c in param_range.T[0]:
            for r in param_range.T[1]:                
                clf = SVC(kernel = 'poly', degree = 2, C=c, coef0=r, gamma='auto', decision_function_shape='ovo')
                cv = cv_performance(clf, X=X, y=y, k=k, metric=metric)
                if(cv > best_performance):
                    best_performance = cv
                    best_C_val = c
                    best_r_val = r


        

    elif(search == 'random'):
        num_samples = 25
        random_exponents = np.random.uniform(low=-2, high=3, size=(num_samples, 2))
        random_params = 10 ** random_exponents
        res = np.zeros(25)

        for c, r in (random_params):
            clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto')
            cv = cv_performance(clf, X=X, y=y, k=5, metric=metric)  
            if(cv > best_performance):
                best_performance = cv
                best_C_val = c
                best_r_val = r


    return best_C_val, best_r_val


def train_word2vec(fname):
    """
    Train a Word2Vec model using the Gensim library.
    First, iterate through all reviews in the dataframe, run your extract_word() function on each review, and append the result to the sentences list.
    Next, instantiate an instance of the Word2Vec class, using your sentences list as a parameter.
    Return the Word2Vec model you created.
    """
    df = load_data(fname)
    sentences = []
    # TODO: Complete this function

    # TODO: Implement this function
    for iter, row in df.iterrows():
        sentence = row['reviewText']
        words = extract_word(sentence)
        sentences.append(words)
            
    model = Word2Vec(sentences, workers=1)
    return model


def compute_association(fname, w, A, B):
    """
    Inputs:
        - fname: name of the dataset csv
        - w: a word represented as a string
        - A and B: sets that each contain one or more English words represented as strings
    Output: Return the association between w, A, and B as defined in the spec
    """
    model = train_word2vec(fname)

    # First, we need to find a numerical representation for the English language words in A and B
    # TODO: Complete words_to_array(), which returns a 2D Numpy Array where the ith row is the embedding vector for the ith word in the input set.
    def words_to_array(set):
        arr = []
        for i, word in enumerate(set):
            word_embedding = model.wv.get_vector(word)
            arr.append(word_embedding)
        return np.array(arr)

    # TODO: Complete cosine_similarity(), which returns a 1D Numpy Array where the ith element is the cosine similarity
    #      between the word embedding for w and the ith embedding in the array representation of the input set
    def cos_similarity(set):
        ret = []
        array = words_to_array(set)
        word_embedding = model.wv.get_vector(w)
        for row in array:
            cos_sim = dot(row, word_embedding)/(norm(row)*norm(word_embedding))
            ret.append(cos_sim)


        return np.array(ret)

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A, and the mean cosine similarity between w and the words in B
    return np.mean(cos_similarity(A)) - np.mean(cos_similarity(B))




def main():
    # Read binary data
    # NOTE: THE VALUE OF dictionary_binary WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED
    #       extract_dictionary, AND THE VALUES OF X_train, Y_train, X_test, AND Y_test
    #       WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED extract_dictionary AND
    #       generate_feature_matrix
    fname = "data/dataset.csv"
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )
    #print('test multi')
    #print(extract_word_multi("It's a test sentence! Does it look CORRECT?"))
    # TODO: Questions 2, 3, 4, 5
    #print('number of reviews')
    #print(np.mean(X_train) * len(X_train[0]))
    #result = (np.vstack((X_train, X_test)))
    #column_sums = np.sum(result, axis=0)
    #print(column_sums)
    #amax = column_sums.argmax()
    #for word in dictionary_binary:
    #    if(dictionary_binary[word] == amax):
    #        print(word)
    #        break
    clf = LinearSVC(loss='hinge', penalty='l2', dual=True, random_state=445, C=0.1)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    

    
    coefficients = clf.coef_[0]
    coef = coefficients.argsort()
    top_positive_indexes = coef[-5:][::-1]
    top_negative_indexes = coef[:5]

    # Sample word to index dictionary (replace this with your actual word_to_index dictionary)
    word_to_index = dictionary_binary

    # Find the top 5 most positive and negative words and their coefficients
    top_positive_words = [(word, coefficients[word_to_index[word]]) for word in word_to_index.keys() if word_to_index[word] in top_positive_indexes]
    top_negative_words = [(word, coefficients[word_to_index[word]]) for word in word_to_index.keys() if word_to_index[word] in top_negative_indexes]
    top_positive_words = sorted(top_positive_words, key=lambda x: x[1], reverse=True)
    top_negative_words = sorted(top_negative_words, key=lambda x: x[1], reverse=True)

    words = []
    coefficients = []
    # Print the top 5 most positive and negative words and their coefficients
    
    for word, coefficient in top_positive_words:
        words.append(word)
        coefficients.append(coefficient)
    for word, coefficient in top_negative_words:
        words.append(word)
        coefficients.append(coefficient)
    # Create a bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(words, coefficients, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Coefficients')
    plt.title('Bar Graph of Words and Coefficients')
    plt.xticks(rotation='vertical')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()

    

# Now, the 'result' array contains both X_train and X_test vertically stacked
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    #C = select_param_linear(X=X_train, y=Y_train, C_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000], metric='accuracy')
    #clf = svm.LinearSVC(penalty = 'l1', dual=False, loss='squared_hinge', random_state = 445, C = 1)
    print('linear')
    #C = select_param_linear(X=X_train, y=Y_train, C_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000], metric='accuracy')
    print(select_param_linear(X=X_train, y=Y_train, k=5, metric='auroc', C_range=[0.001, 0.01, 0.1, 1]))
    #clf.fit(X_train, Y_train)
    #pred = clf.predict(X_test)
    #print(performance(Y_test, pred, 'auroc'))
     
    #3.3
    #plot_weight(X=X_train, y=Y_train, penalty='l1', C_range=[0.001, 0.01, 0.1, 1], loss='squared_hinge', dual=False)

    #c, r = select_param_quadratic(X=X_train, y=Y_train, k=5, metric='auroc', param_range=param_range, search = 'grid')
    #print(c)
    #print(r)
    clf = SVC(kernel='poly', degree=2, C=1000, coef0=1, gamma='auto')
    clf.fit(X_train, Y_train)
    pred = clf.decision_function(X_test)
    print('QUAD')
    print(performance(Y_test, pred, metric='auroc'))
    
    #4.1c
    clf = svm.LinearSVC(penalty = 'l2', dual=True, loss='hinge', random_state = 445, C = .01, class_weight = {-1: 1, 1: 10})
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)

    #4.2a
    clf = svm.LinearSVC(penalty = 'l2', dual=True, loss='hinge', random_state = 445, C = .01, class_weight = {-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    pred = clf.predict(IMB_test_features)
    auroc_pred = clf.decision_function(IMB_test_features)
    print('perf')
    print(performance(IMB_test_labels, pred, metric='sensitivity'))

    #4.4 balanced
    falsePos, truePos, thresholds = roc_curve(IMB_test_labels, auroc_pred)
    roc_auc = auc(falsePos, truePos)
    print(pred)
    #create ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(falsePos, truePos, color='red', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='blue', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for balanced dataset')
    plt.show()

    #4.3a
    C = select_param_linear(X=IMB_features, y=IMB_labels, C_range=[1, 2, 3, 4, 5], metric='f1-score')
    clf = svm.LinearSVC(penalty = 'l2', dual=True, loss='hinge', random_state = 445, C = .01, class_weight = {-1: 10, 1: 5})
    clf.fit(IMB_features, IMB_labels)
    pred = clf.predict(IMB_test_features)
    auroc_pred = clf.decision_function(IMB_test_features)

    #4.3b
    print('perf')
    print(performance(IMB_test_labels, auroc_pred, metric='auroc'))
    
    #4.4

    #imbalanced
    falsePos, truePos, thresholds = roc_curve(IMB_test_labels, auroc_pred)
    roc_auc = auc(falsePos, truePos)
    print(pred)
    #create ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(falsePos, truePos, color='red', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='blue', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for unbalanced dataset')
    plt.show()

    #5.1
    print(count_actors_and_actresses('data/dataset.csv'))
    plot_actors_and_actresses('data/dataset.csv', x_label='rating')
    clf = svm.LinearSVC(penalty = 'l2', dual=True, loss='hinge', random_state = 445, C = .1)
    clf.fit(X_train, Y_train)


    #5.1
    coef = clf.coef_[0]
    actor_ind = dictionary_binary['actor']
    actress_ind = dictionary_binary['actress']
    print(coef[actor_ind])
    print(coef[actress_ind])

    #5.2
    mod = train_word2vec('data/dataset.csv')
    word_embedding = mod.wv.get_vector('actor')
    print(word_embedding.shape)
    sims = mod.wv.most_similar('plot', topn=5)  # get other similar words
    print(sims)

    #5.3
    A = set(['her', 'woman', 'women'])
    B = set(['him', 'man', 'men'])
    print('association')
    print(compute_association(fname='data/dataset.csv', w='talented', A=A, B=B))

    #6     
    print('PART 6')                   
    (
        X_train,
        y_train,
        vectorizer
    ) = get_multiclass_training_data()    

    heldout_features = get_heldout_reviews(vectorizer)
    print('heldout features')
    print(heldout_features.shape)

    #param_range = np.array([ [0.1, 0.1], 
                           #[1, 1], 
                           #[10, 10]])
    #c, r = select_param_quadratic(X=X_train, y = y_train, k =5, metric='accuracy', search='random')
    #print(c)   
    #print(r)   
    #c = select_param_linear(X=X_train, y=y_train, k=5, metric='accuracy', C_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000], loss='hinge', penalty='l2', dual=True)
    #clf = svm.LinearSVC(penalty = 'l2', dual=True, loss='hinge', random_state = 445, C = .01, class_weight = {-1: 1, 1: 1})
    c = select_param_linear(X=X_train, y=y_train, k=5, metric='accuracy', C_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000], loss='hinge', penalty='l2', dual=True)
    print('c')
    print(c)
    clf = SVC(kernel='linear', degree = 1, C=c, decision_function_shape='ovo')
    #clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto', decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    #pred = clf.predict(X_test)
    #print('perf')
    #print(performance(y_test, pred, metric='accuracy'))

    pred = clf.predict(heldout_features)
    generate_challenge_labels(pred, 'sachits')
    


if __name__ == "__main__":
    main()
