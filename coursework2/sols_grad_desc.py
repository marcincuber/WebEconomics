#Author: Marcin Cuber
import numpy as np
import math
"""
    :function related to Linear regressions
"""
#Mean squared error- cost calculation
def cost_linear_reg(X_test, y_test, theta):
    #dot product of input X and theta
    linear_pred = np.dot(X_test, theta)
    sum_erros = sum(np.power((linear_pred - y_test), 2))
    # return error which is sum of errors divided by number of training examples
    error = sum_erros / len(y_test)
    return error

def stoch_linear_calc(X_train, y_train, theta, sing_lambda, conv_tol, num_iters, caculated_iters=0):
    #turn X and y into appropriate format
    y = np.reshape(np.array(y_train), (len(y_train), 1))
    X = np.array(X_train)
    #use cost_linear_reg function which already calculates a single instance
    list_of_errors = [cost_linear_reg(X, y, theta)[0]]

    for val_iter in range(1, num_iters + 1):
        for i in range(0,len(y)):
            product_X_theta = np.dot(X[i, :], theta)[0]
            diff_pred = product_X_theta - y[i]
            #initial slop gradient
            initial_slop = (2 / len(y)) * X[i, :] * diff_pred
            initial_slop = np.reshape(initial_slop, (len(initial_slop), 1))
            #new theta calculated
            theta = theta - (sing_lambda * initial_slop)
        cost_error = cost_linear_reg(X, y, theta)
        list_of_errors.append(cost_error[0])

        #when the difference bettwen errors is negative we print that error has increased
        if (list_of_errors[val_iter]- list_of_errors[val_iter - 1]) > 0:
            print('Error has increased from ', list_of_errors[val_iter - 1], ' to ', list_of_errors[val_iter])

        #when we reach an error which is smaller than tolerance we selected and we at iteration 0
        if caculated_iters == 0 and (list_of_errors[val_iter - 1] - list_of_errors[val_iter]) < conv_tol:
            print('Convergence tolerance= ',conv_tol,' reached, and convereged at iteration: ',val_iter, ' with error: ', list_of_errors[val_iter])
            break

        elif caculated_iters >= 1 and val_iter == caculated_iters:
            print('Already converged at iteration ', val_iter)
            break

    return theta, list_of_errors

def batch_linear_calc(X_train, y_train, theta, sing_lambda, conv_tol, num_iters, caculated_iters=0):
    #turn X and y into appropriate format
    y = np.reshape(np.array(y_train), (len(y_train), 1))
    X = np.array(X_train)
    #use cost_linear_reg function which already calculates a single instance
    list_of_errors = [cost_linear_reg(X, y, theta)[0]]

    for val_iter in range(0,num_iters):
        initial_slop = 0
        for i in range(0,len(y)):
            product_X_theta = np.dot(X[i, :], theta)
            diff_pred = product_X_theta - y[i]
            #initial slop gradient
            initial_slop = initial_slop + ((2 / len(y)) * X[i, :] * diff_pred)

        initial_slop = np.reshape(initial_slop, (len(initial_slop), 1))
        #new theta calculated
        theta = theta - (sing_lambda * initial_slop)

        cost_error = cost_linear_reg(X, y, theta)
        list_of_errors.append(cost_error[0])

        #when the difference bettwen errors is negative we print that error has increased
        if (list_of_errors[val_iter]- list_of_errors[val_iter - 1]) > 0:
            print('Error has increased from ', list_of_errors[val_iter - 1], ' to ', list_of_errors[val_iter])

        #when we reach an error which is smaller than tolerance we selected and we at iteration 0
        if caculated_iters == 0 and (list_of_errors[val_iter - 1] - list_of_errors[val_iter]) < conv_tol:
            print('Convergence tolerance= ',conv_tol,' reached, and convereged at iteration: ',val_iter, ' with error: ', list_of_errors[val_iter])
            break

        elif caculated_iters >= 1 and val_iter == caculated_iters:
            print('Already convereged at iteration ', val_iter)
            break

    return theta, list_of_errors

"""
    :function related to Logistic regressions
"""
#Mean squared error- cost calculation
def cost_logistic_reg(X_test, y_test, theta):
    product_X_theta = -np.dot(X_test, theta)
    logistic_pred = 1/(1 + np.exp(product_X_theta))
    sum_errors = sum(np.power((logistic_pred - y_test), 2))
    # return error which is sum of errors divided by number of training examples
    error = sum_errors / len(y_test)
    return error

def stoch_logistic_calc(X_train, y_train, theta, sing_lambda, conv_tol, num_iters, caculated_iters=0):
    #turn X and y into appropriate format
    X = np.array(X_train)
    y = np.reshape(np.array(y_train), (len(y_train), 1))
    #use cost_logistic_reg function which already calculates a single instance
    list_of_errors = [cost_logistic_reg(X, y, theta)[0]]

    for val_iter in range(1, num_iters + 1):
        for i in range(0,len(y)):
            product_X_theta = -np.dot(X[i, :], theta)
            #product_X_theta = 1/(1 + math.exp(product_X_theta))
            product_X_theta = 1/(1 + np.exp(product_X_theta))
            diff_pred = product_X_theta - y[i]
            #initial slop gradient
            initial_slop = (2 / len(y)) * X[i] * diff_pred
            initial_slop = np.reshape(initial_slop, (len(initial_slop), 1))
            #new theta calculated
            theta = theta - (sing_lambda * initial_slop)

        list_of_errors.append(cost_logistic_reg(X, y, theta)[0])

        #when the difference between errors is negative we print that error has increased
        if (list_of_errors[val_iter]- list_of_errors[val_iter - 1]) > 0:
            print('Error has increased from ', list_of_errors[val_iter - 1], ' to ', list_of_errors[val_iter])

        #when we reach an error which is smaller than tolerance we selected and we at iteration 0
        if caculated_iters == 0 and (list_of_errors[val_iter - 1] - list_of_errors[val_iter]) < conv_tol:
            print('Convergence tolerance= ',conv_tol,' reached, and convereged at iteration: ',val_iter, ' with error: ', list_of_errors[val_iter])
            break

        elif caculated_iters >= 1 and val_iter == caculated_iters:
            print('Already convereged at iteration ', val_iter)
            break

    return theta, list_of_errors

def batch_logistic_calc(X_train, y_train, theta, sing_lambda, conv_tol, num_iters, caculated_iters=0):
    #turn X and y into appropriate format
    X = np.array(X_train)
    y = np.reshape(np.array(y_train), (len(y_train), 1))
    #use cost_logistic_reg function which already calculates a single instance
    list_of_errors = [cost_logistic_reg(X, y, theta)[0]]

    for val_iter in range(1, num_iters + 1):
        initial_slop = 0
        for i in range(0,len(y)):
            product_X_theta = -np.dot(X[i, :], theta)
            #product_X_theta = 1/(1 + math.exp(product_X_theta))
            product_X_theta = 1/(1 + np.exp(product_X_theta))
            diff_pred = product_X_theta - y[i]
            initial_slop = initial_slop + ((2 / len(y)) * X[i] * diff_pred)

        initial_slop = np.reshape(initial_slop, (len(initial_slop), 1))
        theta = theta - (sing_lambda * initial_slop)

        list_of_errors.append(cost_logistic_reg(X, y, theta)[0])

        #when the difference between errors is negative we print that error has increased
        if (list_of_errors[val_iter]- list_of_errors[val_iter - 1]) > 0:
            print('Error has increased from ', list_of_errors[val_iter - 1], ' to ', list_of_errors[val_iter])

        #when we reach an error which is smaller than tolerance we selected and we at iteration 0
        if caculated_iters == 0 and (list_of_errors[val_iter - 1] - list_of_errors[val_iter]) < conv_tol:
            print('Convergence tolerance= ',conv_tol,' reached, and convereged at iteration: ',val_iter, ' with error: ', list_of_errors[val_iter])
            break

        elif caculated_iters >= 1 and val_iter == caculated_iters:
            print('Already convereged at iteration ', val_iter)
            break

    return theta, list_of_errors