## part4.py
# In this file, helper functions are defined for training the simple neural
# network in part 2 using vanilla gradient descent, and plotting learning
# curves.

def part4_gradient_descent(X, Y, init_W, alpha, eps, max_iter):
    '''
    part4_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(W.T, X). The algorithm terminates when successive
    values of theta differ by less than eps (convergence), or when the number of
    iterations exceeds max_iter.
    
    Arguments:
        X -- input data for X (the data to be used to make predictions)
        Y -- input data for X (the actual/target data)
        init_W -- the initial guess for the local minimum (starting point)
        alpha -- the learning rate; proportional to the step size
        eps -- used to determine img[0]when the algorithm has converged on a 
               solution
        max_iter -- the maximum number of times the algorithm will loop before
                    terminating
    '''
    iter = 0
    previous_W = 0
    current_W = init_W.copy()
    firstPass = True
    history = list()
    
    m = Y.shape[1]
    
    # Do-while...
    while(firstPass or
            (np.linalg.norm(current_W  - previous_W) > eps and 
            iter < max_iter)):
        firstPass = False
        
        previous_W = current_W.copy() # Update the previous theta value
        
        # Update theta
        current_W = current_W - alpha * part6_grad_J(current_W, X, Y)
        
        if(iter % (max_iter // 100) == 0):
            # Print updates every so often
            cost = part6_J(current_W, X, Y)
            history.append((iter, cost))
            print("Iter: ", iter, " | Cost: ", cost)
            
        iter += 1
    
    return(current_W, history)
    
def part4_train():
    '''
    part4_train returns...
    '''
    
    
def part4_classify():
    '''
    part4_classify returns...
    '''

def part4_plotLearningCurves(history):
    '''
    part4_plotLearningCurves plots the learning curves associated with training
    a neural network.
    
    Arguments:
        history -- a list of pairs of numbers (error, num_examples), where
                   error is the error associated with training the neural
                   network using num_examples training examples.
    '''
    