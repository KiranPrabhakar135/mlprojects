import Src.LinearRegression as linreg
import Src.LogisticRegression as logreg
import Src.NeuralNetworks as nn
import Test.LogisticRegression as test

m_values = [10]
alpha_array = [0.01]  # [0.03,0.05, 0.07, 0.09]
plot_graphs = False


# Linear Regression
def linear_regression():
    print("------------Linear Regression Started--------------")
    print("Started Linear Regression - Concatenated human setting")
    linreg.con_hum_multiple_alpha(alpha_array)
    if plot_graphs:
        linreg.plot_alpha_vs_accuracy(alpha_array)
        linreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Linear Regression - Concatenated human setting")
    print("Started Linear Regression - Subtracted human setting")
    linreg.sub_hum_multiple_alpha(alpha_array)
    if plot_graphs:
        linreg.plot_alpha_vs_accuracy(alpha_array)
        linreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Linear Regression - Subtracted human setting")
    print("Started Linear Regression - Concatenated GSC setting")
    linreg.con_gsc_multiple_alpha(alpha_array)
    if plot_graphs:
        linreg.plot_alpha_vs_accuracy(alpha_array)
        linreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Linear Regression - Concatenated GSC setting")
    print("Started Linear Regression - Subtracted GSC setting")
    linreg.sub_gsc_multiple_alpha(alpha_array)
    if plot_graphs:
        linreg.plot_alpha_vs_accuracy(alpha_array)
        linreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Linear Regression - Subtracted GSC setting")
    print("------------All Linear Regression's Completed--------------")


# Logistic Regression
def logistic_regression():
    print("------------Logistic Regression Started--------------")
    print("Started Logistic Regression - Concatenated human setting")
    logreg.con_hum_logistic_regression_multiple_alpha(alpha_array)
    if plot_graphs:
        logreg.plot_alpha_vs_accuracy(alpha_array)
        logreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Logistic Regression - Concatenated human setting")
    print("Started Logistic Regression - Subtracted human setting")
    logreg.sub_hum_logistic_regression_multiple_alpha(alpha_array)
    if plot_graphs:
        logreg.plot_alpha_vs_accuracy(alpha_array)
        logreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Logistic Regression - Subtracted human setting")
    print("Started Logistic Regression - Concatenated GSC setting")
    logreg.con_gsc_logistic_regression_multiple_alpha(alpha_array)
    if plot_graphs:
        logreg.plot_alpha_vs_accuracy(alpha_array)
        logreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Logistic Regression - Concatenated GSC setting")
    print("Started Logistic Regression - Subtracted GSC setting")
    logreg.sub_gsc_logistic_regression_multiple_alpha(alpha_array)
    if plot_graphs:
        logreg.plot_alpha_vs_accuracy(alpha_array)
        logreg.plot_alpha_vs_erms(alpha_array)
    print("Finished Logistic Regression - Subtracted GSC setting")
    print("------------All Logistic Regression's Completed--------------")


# Neural Networks
def neural_networks():
    print("------------Neural Network Started--------------")
    print("Started Neural Network - Concatenated human setting")
    nn.con_hum()
    print("Finished Neural Network - Concatenated human setting")
    print("Started Neural Network - Subtracted human setting")
    nn.sub_hum()
    print("Finished Neural Network - Subtracted human setting")
    print("Started Neural Network - Concatenated GSC setting")
    nn.con_gsc()
    print("Finished Neural Network - Concatenated GSC setting")
    print("Started Neural Network - Subtracted GSC setting")
    nn.sub_gsc()
    print("Finished Neural Network - Subtracted GSC setting")
    print("------------All Neural Network Completed--------------")


# linear_regression()
# logistic_regression()
# neural_networks()

# logreg.con_hum_logistic_regression_multiple_alpha(alpha_array)

test.log_reg_test()