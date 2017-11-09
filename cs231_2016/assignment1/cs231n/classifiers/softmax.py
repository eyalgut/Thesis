import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=W.shape[1]
  for i in range(num_train):
    nom=0
    normalizer=0
    for j in range(num_classes):
        normalizer+=np.exp(np.dot(W[:,j],X[i]))
        if j==y[i]:
            nom=np.exp(np.dot(W[:,j],X[i]))
    loss-=np.log(nom/normalizer)
    
    for j in range(num_classes):
        dW[:,j]+=X[i]*np.exp(np.dot(W[:,j],X[i]))/normalizer
        if j==y[i]:
               dW[:,j]-=X[i]  
    
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
    
  dW/=num_train
  dW+=reg*np.sum(W*W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=W.shape[1]
  nom= np.exp(np.sum(W[:,y]*X.T,axis=0))
  normalizer= np.sum(np.exp(np.dot(X,W)),axis=1)
  loss=-np.sum(np.log(nom/normalizer)) 
  
  score_mat=np.exp(np.dot(X,W))/normalizer.reshape(-1,1)
  score_mat[[range(num_train),y]]-=1
  dW=np.dot(X.T,score_mat)

  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW/=num_train
  dW+=reg*np.sum(W*W)
  return loss, dW
