import DataSet.DataProcessing as data
import sklearn.linear_model.logistic as logreg
import numpy as np


def log_reg_test():
    print("test")
    model = logreg.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(data.concatenated_human_training_data,
                                                                                                     data.concatenated_human_training_target_data)

    pred = model.predict(np.array(data.concatenated_human_training_data)[:2, :])
    print(pred)

    prob = model.predict_proba(np.array(data.concatenated_human_training_data)[:2, :]) # doctest: +ELLIPSIS
    print(prob)
    score = model.score(np.array(data.concatenated_human_training_data), np.array(data.concatenated_human_training_target_data))
    print(score)




