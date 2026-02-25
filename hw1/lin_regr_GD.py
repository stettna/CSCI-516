import numpy as np

#This function is the result of the partial derivative with respect to J()

def y(x, w, b):
    return w*x + b

def partial_J(W, X, T, N, j):
    result = 0
    for i in range(0,N):
        result += (y(X[1][i],W[1], W[0]) - T[i])*X[j][i]

    return result/N

def main():
    X  = np.array([[1,1,1,1],[1, 2, 3, 4]])
    T  = np.array([10, 20, 30, 40])

    #number of samples
    N = 4
    D = 2

    W_0 = np.array([1,1])

    alpha = .1
    W_1 = []
    
    for j in range(0, D):
        W_1.append(W_0[j] - alpha*(partial_J(W_0, X, T, N, j)))


    print("W_1 = [", W_1[0], ",",W_1[1], "]" )

main()
    
'''
Output =>  W_1 = [ 3.15 , 7.5 ]
'''