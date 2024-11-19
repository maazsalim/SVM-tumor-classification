
# Author: Swati Mishra
# Created: Sep 23, 2024
# License: MIT License
# Purpose: This python file includes boilerplate code for Assignment 3

# Usage: python support_vector_machine.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added boilerplate code

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value

        #initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
     
        #self.weights = np.zeros(X.shape[1])

        self.weights = np.random.rand(X.shape[1])


    def pre_process(self,):

        #using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input)

        Y_ = self.target 

        return X_,Y_ 
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        loss=0

        # hinge loss implementation- start

        # Part 1

        hinge_loss = np.maximum(0, 1 - Y * np.dot(X, self.weights))
        return 0.5 * np.dot(self.weights, self.weights) + self.C * np.mean(hinge_loss)

        # hinge loss implementatin - end
                
        #return loss
    
    def stochastic_gradient_descent(self, X, Y, X_val, Y_val, part3=False):
        training_losses = []
        validation_losses = []
        previous_loss = float('inf')
        epochs_list = []
        early_stop_epoch = None
        stopping_threshold = 1e-9 if part3 else 1e-4  # Smaller threshold for Part 3

        for epoch in range(self.epoch):
            features, output = shuffle(X, Y)
            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights -= self.learning_rate * gradient

            # Compute training and validation loss
            loss = self.compute_loss(X, Y)
            val_loss = self.compute_loss(X_val, Y_val)
            training_losses.append(loss)
            validation_losses.append(val_loss)
            epochs_list.append(epoch)

            # Check for early stopping condition
            if abs(previous_loss - loss) < stopping_threshold and early_stop_epoch is None:
                early_stop_epoch = epoch  
                if part3:
                    print("Early stopping epoch identified:", epoch)

                print("Early stopping epoch identified:", epoch)
            
            previous_loss = loss

        print("Training ended after reaching maximum epochs.")
        print("Weights:", self.weights)

        # Plot training and validation loss
        if not part3:
            plt.figure()
            plt.plot(epochs_list, training_losses, label='Training Loss')
            plt.plot(epochs_list, validation_losses, label='Validation Loss')
            if early_stop_epoch is not None:
                plt.axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stopping Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        return epochs_list, training_losses, validation_losses



    def mini_batch_gradient_descent(self, X, Y, X_val, Y_val, batch_size):

        training_losses = []
        validation_losses = []


        interval = max(1, self.epoch // 10)

        for epoch in range(self.epoch):
            features, output = shuffle(X, Y)
            for i in range(0, len(features), batch_size):
                X_batch = features[i:i + batch_size]
                Y_batch = output[i:i + batch_size]
                gradients = np.array([self.compute_gradient(X_batch[j], Y_batch[j]) for j in range(len(X_batch))])
                avg_gradient = np.mean(gradients, axis=0)
                self.weights -= self.learning_rate * avg_gradient
                
            current_loss = self.compute_loss(features, output)
            training_losses.append(current_loss)
            val_loss = self.compute_loss(X_val, Y_val)
            validation_losses.append(val_loss)
            
            if epoch % interval == 0:
                print(f"Epoch {epoch}: Training Loss = {current_loss}, Validation Loss = {val_loss}")

        self.plot_losses(training_losses, validation_losses, interval=interval)

    def sampling_strategy(self, X, Y):

        x = X[0]
        y = Y[0]
        
        # Calculate hinge loss
        distances = [1 - y * np.dot(x, self.weights) for x, y in zip(X, Y)]
        
        # Find the sample with the minimum distance (most uncertain sample)
        min_index = np.argmin(distances)
        
        # Select the most uncertain sample based on the minimum distance
        selected_x, selected_y = X[min_index], Y[min_index]
        
        # Return selected sample
        return selected_x, selected_y

    def plot_losses(self, training_losses, validation_losses, stopping_epoch=None, interval=1):

        epochs = range(0, len(training_losses))
        plt.plot(epochs, training_losses, label='Training Loss', color='blue')
        plt.plot(epochs, validation_losses, label='Validation Loss', color='orange')
        
        if stopping_epoch is not None:
            plt.axvline(x=stopping_epoch, color='red', linestyle='--', label='Early Stopping Epoch')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self,X_test,Y_test):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        #compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        #print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision - start
        # Part 2
        precision = precision_score(Y_test, predicted_values)
        #compute precision - end

        #compute recall - start
        recall = recall_score(Y_test, predicted_values)
        #compute recall - end
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        return accuracy

def part_1(X_train, y_train, X_val, y_val):
    my_svm = svm_(learning_rate=0.001, epoch=100, C_value=0.1, X=X_train, Y=y_train)
    X_processed, Y_processed = my_svm.pre_process()
    my_svm.stochastic_gradient_descent(X_processed, Y_processed, X_val, y_val)  
    return my_svm

def part_2(X_train, y_train, X_val, y_val):
    my_svm = svm_(learning_rate=0.001, epoch=500, C_value=0.1, X=X_train, Y=y_train)
    X_processed, Y_processed = my_svm.pre_process()
    my_svm.mini_batch_gradient_descent(X_processed, Y_processed,X_val, y_val, batch_size=24)
    return my_svm

def part_3(X_train, y_train, X_val, y_val):

    C = 0.01
    learning_rate = 0.001
    epoch = 100
  
    # Instantiate the SVM model
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)
    
    # Pre-process the data
    X_train_processed, y_train_processed = my_svm.pre_process()
    
    # Set the initial number of samples for training
    initial_samples = 10
    
    # Lists to store losses and epochs across all iterations for averaging
    training_losses = []
    validation_losses = []
    epochs_list = []  

    cumulative_epoch = 0  # Start from epoch 0
    
    # Train the model iteratively, adding uncertain samples each iteration

    for _ in range(epoch):

        epochs, train_loss, val_loss = my_svm.stochastic_gradient_descent(
            X_train_processed[:initial_samples], y_train_processed[:initial_samples], X_val, y_val, part3=True
        )

        cumulative_epoch+=1
        epochs_list.append(cumulative_epoch)

        # Store the last training and validation loss of each iteration
        training_losses.append(train_loss[-1])
        validation_losses.append(val_loss[-1])

        # Select the most uncertain sample from the remaining data
        x_next, y_next = my_svm.sampling_strategy(X_train_processed[initial_samples:], y_train_processed[initial_samples:])
        
        # Add the selected sample to the training set
        X_train_processed = np.append(X_train_processed, [x_next], axis=0)
        y_train_processed = np.append(y_train_processed, [y_next], axis=0)
        
        # Increase the number of samples 
        initial_samples += 1
        accuracy = my_svm.predict(X_train_processed, y_train_processed)
        if accuracy > 0.92:
            break

    
    # Plot the losses with epochs on the x-axis
    plt.figure()
    plt.plot(epochs_list, training_losses, label='Training Loss')
    plt.plot(epochs_list, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs with Active Learning Iterations')
    plt.legend()
    plt.show()
    
    return my_svm




#Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#normalize the test set separately
scalar = StandardScaler().fit(X_val)
X_Val_Norm = scalar.transform(X_val)

svm_part_one = part_1(X_train, y_train, X_val, y_val)

# testing the model
print("Testing model accuracy...")
svm_part_one.predict(X_Val_Norm,y_val)

svm_part_two = part_2(X_train, y_train, X_val, y_val)

#testing the model
print("Testing model accuracy...")
svm_part_two.predict(X_Val_Norm,y_val)

svm_part_three = part_3(X_train, y_train, X_val, y_val)



