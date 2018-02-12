from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def train_model(train, test, train_features, test_features, class_names, method):
    losses = []
    predictions = {'id': test['id']}
    
    for class_name in class_names:
        train_target = train[class_name]

        if method == 'logreg':
            classifier = LogisticRegression(solver='sag')
        elif method == 'sgd':
            classifier = SGDClassifier(loss='log', penalty='l1', n_jobs=3)
        
        cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        losses.append(cv_loss)
        print('CV score for class {} is {}'.format(class_name, cv_loss))

        classifier.fit(train_features, train_target)
        predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

    mean_auc =  np.mean(losses)
    print('Total CV score is {}'.format(mean_auc))
    predictions = pd.DataFrame.from_dict(predictions)

    return [predictions, mean_auc]
        
