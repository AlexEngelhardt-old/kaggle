import pandas as pd

def preprocess(train, test):
    train_text = train['comment_text']
    test_text = test['comment_text']
    all_text = pd.concat([train_text, test_text])

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # Ideally though, the class_names should be generated automatically

    return [train, test, train_text, test_text, all_text, class_names]
