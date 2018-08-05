# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:44 2018

@author: Keshav Bachu
"""
#Using the tensorflow archetecture for development
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import numpy as np
from copy import copy


#Creates the placeholders for X and Y
#Placeholders represent what will be placed as an input later on when generating the model, unknown number of examples
#which allows one to evaluate as many examples as needed (minus things like memory limitations etc)
def createPlaceholders(X, Y):
    Xshape = X.shape[0]
    Yshape = Y.shape[0]
    Xplace = tf.placeholder(tf.float32, shape = (Xshape, None))
    Yplace = tf.placeholder(tf.float32, shape = (Yshape, None))  
    
    return Xplace, Yplace

#Used to create the variables of float 32 with a given shape
#uses a network shape to create the required number of variables for the hidden layers
#this should not be confused with the placeholders from X and Y 
def createVariables(networkShape):
    placeholders = {}
    
    for i in range(1, len(networkShape)):
        placeholders['W' + str(i)] = tf.get_variable(name = 'W' + str(i), shape=[networkShape[i], networkShape[i - 1]], initializer=tf.contrib.layers.xavier_initializer())
        placeholders['b' + str(i)] = tf.get_variable(name = 'b' + str(i), shape=[networkShape[i], 1], initializer=tf.zeros_initializer())
    return placeholders

def setVariables(weightsExist):
    placeholders = {}
    
    for key in weightsExist:
        placeholders[key] = tf.get_variable(name = key, initializer=weightsExist[key])
    return placeholders


#Forward propogation using a, relu and sigmoid function on respective layers (relu 1 - n-1 layer and sigmoid for n layer)
def forwardProp(X, placeholders, networkShape, keep_probability = None):
    #total number of parameters in the network, divided by 2 for the number of layers within it with X being 0
    totalLength = len(placeholders)/2
    totalLength = int(totalLength)
    
    val1 = X
    val2W = placeholders['W' + str(1)]
    val2b = placeholders['b' + str(1)]
    
    pass_Z = tf.matmul(val2W, val1) + val2b
    pass_A = tf.nn.relu(pass_Z)
    
    for i in range (1, totalLength):
        val_W = placeholders['W' + str(i + 1)]
        val_b = placeholders['b' + str(i + 1)]
        
        pass_Z = tf.matmul(val_W, pass_A) + val_b
        pass_A = tf.nn.relu(pass_Z)
        #if(keep_probability != None and i != totalLength - 1):
            #pass_A = tf.nn.dropout(pass_A, keep_probability)
    
    return pass_Z

#Cost function for this sigmoid network
def computeCost(finalZ, Y, weights = None):
    logits = tf.transpose(finalZ)
    labels = tf.transpose(Y)
    
    length = len(weights)
    reg_lambda = 0.01
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    #regularizationL2 = lambda/(2 * M) * norm(w) ** 2
    #regularization = tf.nn.l2_loss(weights['W' + str( int(length/2) - 1)]) + tf.nn.l2_loss(weights['W' + str( int(length/2))]) + tf.nn.l2_loss(weights['W' + str( int(length/2) - 2)])
    #cost = tf.reduce_mean(cost + reg_lambda * regularization)
    return cost

#Training the model, X and Y inputs for training and testing NN
#Network shape - dictates the shape of the network; given as a list
#Learning rate - step size of backprop
#iterations -  Number of iterations of NN
#print_cost - controles if cost is printed every 100 iterations
def trainModel(xTest, yTest,netShape, xDev = None, yDev = None,  learning_rate = 0.0001, itterations = 1500, print_Cost = True, weightsExist = None, minibatchSize = 1):
 
    ops.reset_default_graph()
    costs = []                      #used to graph the costs at the end for a visual overview/analysis
    networkShape = copy(netShape)
    
 
    #Need to first create the tensorflow placeholders
    Xlen = xTest.shape[0]   #get the number of rows, aka features for the input data
    networkShape.insert(0, Xlen)
    
    X, Y = createPlaceholders(xTest, yTest)
    keep_probability = tf.placeholder(tf.float32)
    
    if(weightsExist == None):
        placeholders = createVariables(networkShape)
    else:
        placeholders = setVariables(weightsExist)
    
    #define how Z and cost should be calculated
    Zfinal = forwardProp(X, placeholders, networkShape, keep_probability)
    cost = computeCost(Zfinal, Y, placeholders)
    #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
    #Set global variables and create a session
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #temp_cost = 0 
        for itter in range(itterations):
            _,temp_cost = sess.run([optimizer, cost], feed_dict={X:xTest, Y: yTest, keep_probability: 1})
            
            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
                
            if(itter % 5 == 0):
                costs.append(temp_cost)
            
            
        parameters = sess.run(placeholders)
        Youtput = Zfinal.eval({X: xTest, Y: yTest, keep_probability: 1})
        #tf.eval(Zfinal)
        
        #confidance level of 75% can be adjusted later
        prediction = tf.equal(tf.greater(Zfinal, tf.constant(0.5)), tf.greater(Y, tf.constant(0.5)))
        predOut = prediction.eval({X: xTest, Y: yTest, keep_probability: 1})
        
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest, keep_probability: 1}))
        #print ("Test Accuracy:", accuracy.eval({X: xDev, Y: yDev}))
        plt.plot(costs)
        
    return parameters, Youtput, predOut.astype(int)


def predictor(weights, networkShape, xTest, yTest):
    ops.reset_default_graph()
    
    networkShape2 = copy(networkShape)
    Xlen = xTest.shape[0]
    networkShape2.insert(0, Xlen)
    
    X, Y = createPlaceholders(xTest, yTest)
    keep_probability = tf.placeholder(tf.float32)
    
    placeholders = setVariables(weights)
    Zfinal = forwardProp(X, placeholders, networkShape2, keep_probability)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        Youtput = Zfinal.eval({X: xTest, Y: yTest, keep_probability: 1}) 
        
        prediction = tf.equal(tf.greater(Zfinal, tf.constant(0.5)), tf.greater(Y, tf.constant(0.5)))
        predOut = prediction.eval({X: xTest, Y: yTest, keep_probability: 1})
        
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        checkVector = prediction.eval({X: xTest, Y: yTest, keep_probability: 1})
        print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest, keep_probability: 1}))
    
    return Youtput, checkVector, predOut.astype(int)

