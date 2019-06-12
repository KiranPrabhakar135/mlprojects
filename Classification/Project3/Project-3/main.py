import LogisticRegression as log_reg
import SupportVectorMachine as svm
import NeuralNetworks as nn
import RandomForest as rf
import MaxVoting as ensemble_classifier


def perform_classifications():
    print("........................................")
    print(".........Kprabhak - 50287403............")
    print("........................................")
    print("............Classification..............")
    print(".......Started Softmax Regression.......")
    log_reg.calculate_weights_perform_logistic_regression_and_output_accuracy()
    print(".......Started SVM Classification.......")
    svm.create_model_perform_classification_and_output_accuracy()
    print("..Started Neural Network Classification..")
    nn.create_model_perform_classification_and_output_accuracy()
    print("....Started Random Forests Classification....")
    rf.create_model_perform_classification_and_output_accuracy()
    print("..........Started Max Voting............")
    ensemble_classifier.perform_max_voting_classification()


perform_classifications()