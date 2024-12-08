# add your code to this file
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ =="__main__":
    # check if the argument is valid
    if len(sys.argv) < 2:
        print("Please provide a filename to read.")
        sys.exit()
        
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    #Q1
    #Read data from csv file 
    years = []
    frozen_days = []

    # read data and convert into arrays
    with open(filename, 'r') as file:
        next(file)
    
        for line in file:
            year, days = line.split(',')
            years.append(int(year))
            frozen_days.append(int(days))

    file.close()

    #Q2
    #Produce the plot from read data
    plt.plot(years, frozen_days)
    plt.title('Year vs. Number of Frozen Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("data_plot.jpg")
    # plt.show()
    plt.close()

    #Q3
    #Compute the matrix X from given data    
    n1 = len(years) 
    
    # initialize the X matrix 
    X = np.zeros((n1, 2), dtype = np.float64)
    
    # allocate x values into the matrix
    X_min = min(years)
    X_max = max(years)
    for i in range(n1):
        X[i] = [((years[i] - X_min) / (X_max - X_min)), 1]
    X_normalized = X

    print("Q3:")
    print(X_normalized)

    #Q4
    #Compute the corresponding y values into a vector Y
    n2 = len(frozen_days)
    
    # initialize the Y matrix 
    Y = np.zeros(n2, dtype = np.int64)
    
    # allocate y values into the matrix
    for i in range(n2):
        Y[i] = frozen_days[i]
    
    #print("Q4a:")
    #print(Y)
    
    #Compute the matrix product X^TX (denoted as Z)  
    Z = np.dot(X.T, X)
    
    #print("Q4b:")
    #print(Z)

    #Compute the inverse of Z (denoted as I)
    I = np.linalg.inv(Z)
    
    #print("Q4c:")
    #print(I)

    #Compute the pseudo-inverse of X (denoted as PI)
    PI = np.dot(I, X.T)
    
    #print("Q4d:")
    #print(PI)

    #Compute the weights from results we previouly calculated (weights = PI * Y)
    weights = np.dot(PI, Y)
    cf_weights = weights #closed form solution weights
    
    print("Q4:")
    print(weights)

    #Q5
    # Algorithm 1
    def algorithm_1(X, Y, learning_rate, iterations, print_weights = False):
        #Compute the hat_y from the given weights
        weights = np.zeros(2, dtype = np.float64)

        for t in range(0, iterations - 1):
            if t % 10 == 0 and print_weights:
                print(weights) #Q5a

            gradient = np.zeros(2, dtype = np.float64)
            Y_hat = np.zeros(2, dtype = np.float64)
                
            for i in range(0, len(Y)):
                Y_hat = np.dot(weights, X[i])
                error = Y_hat - Y[i]
                gradient += np.dot(error, X[i])
            weights -= gradient / len(Y) * learning_rate
        return weights

    print("Q5a:")
    algorithm_1(X_normalized, Y, learning_rate, iterations, print_weights = True)
    
    #Q5b/c - Find the optimal learning rate and iterations
    # lr = 0.01 # 0.7289 at 100 iterations | 0.17737 at 389 | .383376 at 200
    # i = 100
    # while True:
    #     weights = algorithm_1(X_normalized, Y, lr, i, print_weights = False)
    #     if np.allclose(weights, cf_weights, rtol = 0, atol = 0.01):
    #         print("Q5b:", lr)
    #         print("Q5c:", i)
    #         break
    #     elif lr > 0.4 or i > 500:
    #         break
    #     else:
    #         lr *= 1.05
    #         i += 3
    #         print("(LR, diff) : (", lr, np.linalg.norm(weights - cf_weights), ")")
    print("Q5b:", 0.23839900559179253)
    print("Q5c:", 295)

    #Q5d
    # plot MSE loss vs. iterations
    def plot_MSE_loss(X, Y, learning_rate, iterations):
        weights = np.zeros(2, dtype = np.float64)
        losses = {}
        for t in range(0, iterations - 1):
            gradient = np.zeros(2, dtype = np.float64)
            Y_hat = np.zeros(2, dtype = np.float64)
            loss = 0
            for i in range(0, len(Y)):
                Y_hat = np.dot(weights, X[i])
                error = Y_hat - Y[i]
                gradient += np.dot(error, X[i])
                loss += error ** 2
            weights -= gradient / len(Y) * learning_rate
            total_loss = loss / (2 * len(Y))
            losses[t] = total_loss

        plt.plot(losses.keys(), losses.values())
        plt.title('MSE Loss vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('MSE Loss')
        plt.savefig("loss_plot.jpg")
        # plt.show()
        plt.close()
    
    plot_MSE_loss(X_normalized, Y, learning_rate, iterations)


    #Q6
    #Predict the number of ice days for 2022 - 2023 winter
    weights = cf_weights
    x_test = (2023 - X_min) / (X_max - X_min)
    y_hat_test =  weights[1] + (weights[0] * x_test)
    
    print("Q6: " + str(y_hat_test))

    #Q7
    #Interpret the model by the sign of weights[0] (w)
    #Print the symbol by checking whether weights[0] is positive, negative or zero
    if weights[0] > 0:
        symbol = '>'
    elif weights[0] < 0:
        symbol = '<'
    elif weights[0] == 0:
        symbol = '='
    
    print("Q7a: " + symbol)
    
    # Print the interpretation
    print("Q7b: " + "'>' sign indicates that the predictor variable (year) is positively correlated with the number of frozen days, as is they increase," + 
                    " '<' sign indicates that the number of frozen days is likely to decrease," + 
                    " and '=' sign indicates that the number of frozen days remains same on Lake Mendota.")
    
    #Q8
    #Predict the year x_stat with given MLE(Maximum Likelihood Estimation) weights  
    # 0 = beta_hat[0] + beta_hat[1] * x_star
    x_star = (-weights[1] / weights[0]) * (X_max - X_min) + X_min

    print("Q8a: " + str(x_star))
    print("Q8b: " + "The year " + str(x_star) + " is the year when the number of frozen days is expected to be zero on Lake Mendota. " + 
                    "Luckily, I will not have to experience this sad day, as I will be long gone by then. " +
                    "But, it is a sign of global warming and climate change, which is a serious issue that we need to address. " +
                    "If anything, I would hypothesize that the model does not capture the concurrent effects of climate change and that the lake may not freeze over sooner than 2463.")