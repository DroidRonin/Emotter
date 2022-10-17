# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:52:26 2020

@author: Nerv
"""
from tweetclass import *


class Perceptron(object):
    
    def __init__(self, train_tweets, emotion, weights = None, bias = None):    #Constructor that initializes all the attributes
        

        self.train_tweets = train_tweets
        self.emotion = emotion
        self.weights = weights
        self.bias = bias
       # self.epochs = epochs
        

        
    def training(self):                          #Training method
        #emotion = 'Anger' 
        lr = 0.3                                 #This sets the learning rate to 0.3
        #epochs = 5
        wt_mat = []                              #Creating an empty list of the weights
        new = {}                                 
        #b = 0
        self.weights = {}                        #Creating our weights dictionary
        feature_list = []                        #Creating a list for all the features
        b = 0                                    #Initializing bias with 0
        
        for index in self.train_tweets:
            for feature in index.features:
                feature_list.append(feature)  #This appends all the features extracted from tweets to feature_list
                
        for f in feature_list:
            self.weights[f] = 1                             #This initializes all the weights to 1
            
        for i in range(0, len(self.train_tweets)):
            f = []
            f = f + [1] * len(self.train_tweets[i].features)   #This creates an input vector of all the features
            w = f                                              #Setting the weight variable to our input vector
            
            if self.emotion in self.train_tweets[i].emotions:   #This sets the output emotion class to binary values of 0s and 1s
                out = 1
            else:
                out = 0
                
            if(sum(x*y for x,y in zip(f,w)) >= b):          #A dot product of input and features that gives y prediction as 1 if >= bias and 0 if otherwise
                y_pred = 1
            else:
                y_pred = 0
                
            lr_f = [index * lr for index in f]   #multiplies learning rate and feature list
            w_sum = [a + b for a, b in zip(w, lr_f)]   #Adds the weights with lr*input 
            w_diff = [a - b for a, b in zip(w, lr_f)]  #Subtracts weights from lr*input
            
            if out == 1 and y_pred == 0:            #condition that matches the values between the actual output and predicted output and updates the weights and bias. 
                w = w_sum
                b = b - lr * 1
                
            elif out == 0 and y_pred == 1:
                w = w_diff
                b = b + lr * 1
                
            wt_mat.append(w)                       #Appending updated weights to our wt_mat list
            self.bias = b
                      #List of new bias
            for k in wt_mat:                       #Iterates over the wt_mat list and creates a sub dictionary of features and weights of the current index
                new = dict(zip(self.train_tweets[i].features, k))
                self.weights.update(new)    #This updates our main dictionary to new weights
            
                
             
                
