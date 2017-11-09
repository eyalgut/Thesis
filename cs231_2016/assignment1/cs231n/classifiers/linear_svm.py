import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    gard_i=np.zeros(num_classes)
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        gard_i[j]=1
        gard_i[y[i]]-=1
        loss += margin    
    #back prop
    term=X[i].reshape(1,-1)*gard_i.reshape(-1,1) #broadcasting
    term=term.T
    dW_curr=term+reg*W
    dW+=dW_curr
  
  dW/=num_train #go in the average direction
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]   


  scores = X.dot(W)
  correct_class_score = scores[[range(num_train),y]]
  gard=np.zeros((num_train,num_classes))
  margin = scores - correct_class_score.reshape(-1,1) + 1 # note delta = 1

  margin[[range(num_train),y]]=0 #do not include correct class indices
  margin[margin<0]=0  
  loss = np.sum(margin)        
  inds=margin > 0
  gard[inds]=1
  
  margin[margin>0]=1  
  gard[[range(num_train),y]]=-1*np.sum(margin,axis=1)             
  term=np.dot(X.T,gard) 
  dW=term/num_train+reg*W
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
