#Author: Marcin Cuber
#
# Sources used: logistic regression based on code and explanation provide on:
# http://bryantravissmith.com/2015/12/29/implementing-logistic-regression-from-scratch-part-2-python-code/
import sols_grad_desc
import numpy as np
import random
import matplotlib.pyplot as plt
import time
# load spambase data
data = np.genfromtxt("spambase.data",delimiter=",")

# number of folds- k = 10, 10-fold cross validation
fold_num = 10
# get the shape of entire data set (spam_data)
num_rows, num_columns = data.shape

#list of k sub-lists storing data for each fold
folds_sub_lists = [[] for num in range(0,fold_num)]

# placing the indices in their according sub-arrays
for i in range(0,num_rows):
    folds_sub_lists[i % fold_num].append(i)

#randomise data (emails) in each fold so data spam and none-spam are grouped together
for single_list in folds_sub_lists:
    random.shuffle(single_list)

#select last column of imported data
#indicates which emails spam or not
# Y_vector does not need to be normalised
Y_vector = data[:, -1]

#rest of the data apart form Y_vector
#includes all the features
X_matrix = data[:, 0:-1]

#calculat mean and standard deviation for z-score normalisation
all_means = np.mean(X_matrix,axis=0)
all_std = np.std(X_matrix, axis=0)

#do z-score normalisation which has calculation:
# (x- mean)/std
X_matrix_minus_mean = np.subtract(X_matrix, all_means)

#X is now normalised
X_matrix = np.divide(X_matrix_minus_mean, all_std)

# the spam data split into k_fold groups
folds_spam_data = []
for row_val in folds_sub_lists:
    folds_spam_data.append(X_matrix[row_val, :])

"""
    :function running cross validation- 10 folds
"""
def cross_val(fold_num, lambda_list, num_iters, k_fold_data, y_data, k_fold_indices, conv_tol):
    #number of lambda parameters used
    param_num = len(lambda_list)

    # initilise lists of lists which correspond to individual lambda value
    # for the assignment purpose we only need to produce a graph with three different learning curves
    all_y_test = [[] for i in range(0,param_num)]
    all_comp_pred = [[] for i in range(0,param_num)]
    errors_from_test = [[] for i in range(0,param_num)]
    #list to store errors used later for plots
    training_errors_list = [[] for i in range(0,param_num)]
    iterations_single_lambda = []
    for i in range(0,param_num):
        iterations_single_lambda.append(num_iters)

    # training for all folds (10 folds)
    for fold_int in range(0, fold_num):
        for lambda_value in range(0,param_num):
            sel_lambda = lambda_list[lambda_value]
            print('Current fold is: ', (fold_int + 1), ' for lambda: ', sel_lambda, ' ---------------------------' )
            # form training, test sets for Data itself dedicated X_train and X_test
            # first 9 parts is training and remaining is test set
            train_parts = np.arange(len(k_fold_data))
            train_parts = np.delete(train_parts, fold_int)
            # same as above but for y_train and y_test
            train_parts2 = np.arange(len(k_fold_indices))
            train_parts2 = np.delete(train_parts2, fold_int)

            # select 9 folds from 10 and same for y_train
            X_train = []
            for part in train_parts:
                # k-fold partitions together
                X_train.extend(k_fold_data[part])
            #the last set not included in training is our test set
            X_test = k_fold_data[fold_int]

            y_train = []
            for part in train_parts2:
                # form a vector with y values correspoding to each row taken in X sets
                y_train_temp = [y_data[sing_row] for sing_row in k_fold_indices[part]]
                y_train.extend(y_train_temp)
            #the last y value left form a y_test vector
            y_test = [y_data[sing_row] for sing_row in k_fold_indices[fold_int]]

            # we turn X_train, X_test, y_train, y_test into formats that we can use in our calculations
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.reshape(np.array(y_train), (len(y_train), 1))
            y_test = np.reshape(np.array(y_test), (len(y_test), 1))

            #size of training set
            size = len(X_train[0])
            #initilise theta values to zero- starting values
            theta = np.zeros((size, 1))

            #Assign method to calculate gradient descent
            theta, train_errors= sols_grad_desc.stoch_logistic_calc(X_train, y_train, theta, sel_lambda, conv_tol, num_iters, iterations_single_lambda[lambda_value])

            #calculate error on test set
            test_error = sols_grad_desc.cost_logistic_reg(X_test, y_test, theta)
            test_error = test_error[0]

            #calculation required to be used in ROC calculations
            #we need to compute predictions for y vectores
            comp_pred = -np.dot(X_test, theta)
            comp_pred = 1 + np.exp(comp_pred)
            comp_pred = (1 / comp_pred)
            all_comp_pred[lambda_value].extend(comp_pred)
            all_y_test[lambda_value].extend(y_test)

            # test error is need to determine the best lambda for the ROC curve
            errors_from_test[lambda_value].append(test_error)

            training_errors_list[lambda_value].append(train_errors)

    #average test errors for all folds and select minimum
    errors_from_test = np.mean(errors_from_test, axis=1).tolist()
    #minimum error value
    minimum_error = min(errors_from_test)
    #minimum error index corresponding to a position in the list
    minimum_index = errors_from_test.index(minimum_error)

    # take first element from each list
    minimum_comp_pred = np.array([single_list[0] for single_list in all_comp_pred[minimum_index]])

    minimum_y_test = np.array([single_list[0] for single_list in all_y_test[minimum_index]])

    return np.array(training_errors_list), minimum_comp_pred, minimum_y_test

"""
    :Function ploting the curve among with AUC value
"""
def AUC_curve(min_pred,min_y_test):

    enumerator = (1 / 0.01) + 1

    false_rates = []
    true_rates = []
    size_y_test = len(min_y_test)

    #we use values between 0 and 1 and check spam and non spam values against them
    for pos_val in np.linspace(0, 1, enumerator):
        values_spam = []
        values_not_spam = []

        for tuple in zip(range(0,size_y_test), min_y_test):
            if tuple[1] == 1:
                values_spam.append(tuple[0])
            if tuple[1] == 0:
                values_not_spam.append(tuple[0])

        size_spam = len(values_spam)
        size_not_spam = len(values_not_spam)

        pred_spam = []
        pred_not_spam = []

        for value in min_pred[values_spam]:
            if value > pos_val:
                pred_spam.append(value)

        for value in min_pred[values_not_spam]:
            if value > pos_val:
                pred_not_spam.append(value)

        size_pred_spam = len(pred_spam)
        size_pred_not_spam = len(pred_not_spam)

        # x coordinates are false positives
        false_rates.append(size_pred_not_spam / size_not_spam)
        # y coordinates are true positves
        true_rates.append(size_pred_spam /size_spam)

    # sort the x and y coordinates so the AUC can be calculated properly
    false_rates.sort()
    true_rates.sort()

    # we compare y value against predictied ones
    # and we create curve one curve for all folds
    sum_all_AUC = 0

    for i in range(1, len(false_rates)):
        difference = false_rates[i] - false_rates[i - 1]
        sum_ys = true_rates[i] + true_rates[i - 1]
        sum_all_AUC = sum_all_AUC + (difference * sum_ys)

    final_auc = sum_all_AUC/2

    print('The Area Under Curve is: ', final_auc)

    ax = plt.subplot(111)
    #we plot y- true positives against x- false positives
    ax.plot(false_rates, true_rates,color='r', label = ('ROC-Curve with AUC =' + str(final_auc)))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.show()

#STOCHASTIC LOGISTIC REGRESSION
def main():
    #convergence tolerance
    conv_tol = 0.0001

    #select values for each learning paramenter- lambda
    lambda1 = 1
    lambda2 = 0.1
    lambda3 = 0.01

    #lambda - different learning rate parameters
    lambda_list = [lambda1, lambda2, lambda3]
    num_iters = 20

    training_errors, minimum_comp_pred, minimum_y_test = cross_val(fold_num, lambda_list, num_iters, folds_spam_data, Y_vector, folds_sub_lists, conv_tol)

    training_errors_list1 = []
    training_errors_list2 = []
    training_errors_list3 = []

    for lambda_index in range(0,len(lambda_list)):
        if lambda_index == 0:
            errors = training_errors[lambda_index]
            mean_errors = np.mean(errors, axis=0)
            training_errors_list1.extend(mean_errors)
        elif lambda_index == 1:
            errors = training_errors[lambda_index]
            mean_errors = np.mean(errors, axis=0)
            training_errors_list2.extend(mean_errors)
        elif lambda_index == 2:
            errors = training_errors[lambda_index]
            mean_errors = np.mean(errors, axis=0)
            training_errors_list3.extend(mean_errors)
        else:
            print ("No more than 3 lambdas can be used, please modify your inputs")

    training_iteration_list1 = range(0, len(training_errors_list1))
    training_iteration_list2 = range(0, len(training_errors_list2))
    training_iteration_list3 = range(0, len(training_errors_list3))

    # generating plot for all lambdas so we have 3 different test errors
    # all errors are means calculated based on 10 folds
    #print('Here is our figure with lines corresponding to each lambda value')
    ax = plt.subplot(111)

    ax.plot(training_iteration_list1, training_errors_list1,  '-',color='r', label=('lambda='+str(lambda1)))
    ax.plot(training_iteration_list2, training_errors_list2, '--',color='b', label=('lambda='+str(lambda2)))
    ax.plot(training_iteration_list3, training_errors_list3, ':',color='g',label=('lambda='+str(lambda3)))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
    ax.set_xlabel('iteration number')
    ax.set_ylabel('MSE- cost')
    plt.show()

    #function calculating Area Under Curve and displays graph
    #for the curve we use minimum test errors for specific lambda
    AUC_curve(minimum_comp_pred ,minimum_y_test)

t0 = time.clock()
main()
print ((time.clock() - t0), " seconds process time")