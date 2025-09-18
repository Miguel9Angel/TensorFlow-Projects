import numpy as np
import random

class Network():
    def __init__(self, layers, seed=42, cost='quadratic', lmbda=0, initializer='', n_early_stop=0):
        np.random.seed(seed)       
        self.layers = layers
        self.biases = [np.zeros((1, y)) for y in self.layers[1:]]
        self.n_early_stopping = n_early_stop

        if initializer=='Gaussian':
            self.weights = [np.random.randn(x, y)/np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]
        else:
            self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
        
        self.lmbda = float(lmbda)
            
        cost_functions = {
            'quadratic': self.quadratic_cost, 
            'cross_entropy': self.cross_entropy_cost
        }

        if cost.lower() not in cost_functions:
            available = ', '.join(cost_functions.keys())
            raise ValueError(f"Cost function '{cost}' are not available: {available}")
        
        self.cost = cost_functions[cost.lower()]

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def dev_sig(self, a):
        return a*(1-a)

    def quadratic_cost(self, y_true, y_pred, derivate=False):
        if derivate:
            return (y_pred - y_true) * self.dev_sig(y_pred)
        else:
            m = y_true.shape[0]
            return (1/(2*m))*np.sum((y_pred - y_true)**2)
    
    def cross_entropy_cost(self, y_true, y_pred, derivate=False):
        if derivate:
            return y_pred-y_true
        else:
            #np.clip forces the y_pred to be greater than epsilon and less than (1-epsilon) to avoid inf values in np.log
            epsilon = 1e-15  
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    #schedule learning rate, reduce the lr by  halft each 20 epochs 
    def step_lr(self, epoch, lr):
        return lr * 0.5 if epoch % 20 == 0 and epoch > 0 else lr

    #Stocastic gradient descent
    def SGD(self, training_data, epochs=10, batch_size=32, lr=0.1, validation_data=None, return_training_cost=False, step_lr = False):
        n = len(training_data)
        best_accuracy = 0
        counter_stopping = 0
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for epoch in range(epochs):
            random.shuffle( training_data )
            batches = [ training_data [i:i+ batch_size ] for i in range(0, n, batch_size)]
            
            if step_lr:
                lr = self.step_lr(epoch, lr)

            for batch in batches:
                X_batch = np.vstack([x for x, _ in batch])
                y_batch = np.vstack([y for _, y in batch])

                activations = self.feedforward(X_batch)
                grads_w, grads_b = self.backpropagation(activations, y_batch, X_batch.shape[0])
                self.update_w_b(grads_w, grads_b, lr, X_batch.shape[0])

            train_cost, train_accuracy = self.evaluate(training_data)  
            eval_cost, eval_accuracy = self.evaluate(validation_data)
            training_cost.append(train_cost)
            training_accuracy.append(train_accuracy)
            evaluation_cost.append(eval_cost)
            evaluation_accuracy.append(eval_accuracy)

            if self.n_early_stopping > 0 and validation_data:
                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    counter_stopping = 0
                else:
                    counter_stopping += 1
                    
                if counter_stopping >= self.n_early_stopping:
                    print("Early stopping: no improvement in {} epochs".format(self.n_early_stopping))
                    break

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
        
    def feedforward(self, X):
        activations = [X]
        a = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W)+b
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backpropagation(self, activations, y, batch_size):
        n = len(self.weights)
        grads_w = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        delta = self.cost(y, activations[-1], derivate=True)
        
        grads_w[-1] = np.dot(activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)
        
        for idx in range(n-2, -1,-1):
            delta = np.dot(delta, self.weights[idx+1].T)*self.dev_sig(activations[idx+1])
            grads_w[idx] = np.dot(activations[idx].T, delta)
            grads_b[idx] = np.sum(delta, axis=0, keepdims=True)

        if self.lmbda > 0:
            for i in range(len(grads_w)):
                grads_w[i] += (self.lmbda / batch_size) * self.weights[i]
                
        return grads_w, grads_b

    def update_w_b(self, grads_w, grads_b, lr, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= (lr/batch_size)*grads_w[i]
            self.biases[i] -= (lr/batch_size)*grads_b[i]
            
    def cost_function(self, y_pred, y):
        cost = self.cost(y, y_pred, derivate=False)
        if self.lmbda > 0:
            total_params = sum(W.size for W in self.weights)
            l2_penalty = (self.lmbda / (2 * total_params)) * sum(np.sum(W**2) for W in self.weights)
            cost += l2_penalty
        return cost  
        
    def predict(self, X):
        if isinstance(X, list) and isinstance(X[0], tuple):
            X = np.vstack([xi for xi, _ in X])
        activations = self.feedforward(X)
        return activations[-1]

    def evaluate(self, test_data, threshold=0.5):
        X = np.vstack([x for x,_ in test_data])
        y = np.vstack([y for _,y in test_data])
        preds = self.predict(X)
        cost = self.cost_function(preds, y)
        
        if preds.shape[1] == 1:
            preds_bin = (preds >= threshold).astype(int)
            acc = np.mean(preds_bin == y)
        else:
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(preds, axis=1)
            acc = np.mean(y_pred == y_true)
        return cost, acc 
