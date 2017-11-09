import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)



def sig_dx(x):
    #return np.exp(-x)/(1+np.exp(-x))**2
    return sigmoid(x) * (1 - sigmoid(x))
               
               
def tanh_dx(x):
    #return 4/(np.exp(2*x)+np.exp(-2*x)+2)
    return (1 - np.tanh(x)**2)
"""
def dtanh/dx(x):
    ( (np.exp(x)+np.exp(-x))**2-(np.exp(x)-np.exp(-x))**2 )/(np.exp(x)+np.exp(-x)**2
     1-(np.exp(x)-np.exp(-x))**2/(np.exp(x)+np.exp(-x)**2 
     1-((a-b)(a-b))/ ((a+b)(a+b) =1-(a**2-2ab+b**2)/(a**2+2ab+b**2)=1-(a**2-2+b**2)/(a**2+2+b**2)=1-(a**2-2+2-2+b**2)/(a**2+2+b**2)=
     1-(1-4/(a**2+2+b**2))= 4/(a**2+2+b**2)=4(exp(2x)+exp(-2x)+2)**-1 
     
     x=WX*X+b+WH*H
     dx/db=1
     dtanh/dx * dx/db =4(exp(2x)+exp(-2x)+2)**-1 
"""
def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  term_input=np.dot(x,Wx)+b #R^{N,H}, bias is ony for the input weights
  term_prevstate=np.dot(prev_h,Wh) #R^{N,H}
  tanh_input=term_prevstate+term_input
  next_h=np.tanh (tanh_input) #doing an acivation on the sum of inputs
    
  cache=(x,prev_h,Wh,Wx,tanh_input)
  ##############################################################################
  #                             END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  (x,prev_h,Wh,Wx,tanh_input)=cache
  (N, D)=x.shape
  (N,H)=prev_h.shape
  dx, dprev_h, dWx, dWh, db = None,None,None,None,None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x,prev_h,Wh,Wx,tanh_input)=cache
    
 
  dWx=np.dot(x.T,tanh_dx(tanh_input)*dnext_h)#R^{D,H}
  dx=np.dot(tanh_dx(tanh_input)*dnext_h,Wx.T)#R^{N,D}
  dWh=np.dot(prev_h.T,tanh_dx(tanh_input)*dnext_h)#R^{D,H}
  dprev_h=np.dot(tanh_dx(tanh_input)*dnext_h,Wh.T)#R^{D,H}
  db=np.sum(tanh_dx(tanh_input)*dnext_h,axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, []
  (N, T, D)=x.shape
  (D, H)=Wx.shape
  h=np.zeros((N, T, H))
  for i in range(T):
        h0,C=rnn_step_forward(x[:,i,:], h0, Wx, Wh, b)
        h[:,i,:]=h0
        cache.append(C)
        
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  #dx, dh0, dWx, dWh, db = None, None, None, None, None
  (N, T, H)=dh.shape
  #init
  dx=np.zeros((N, T, cache[0][0].shape[1]))
  dx[:,T-1,:], dh0, dWx, dWh, db=rnn_step_backward(dh[:,T-1,:], cache.pop())
  for i in reversed(range(0,T-1)):
        dx[:,i,:], dh0, dWx_, dWh_, db_=rnn_step_backward(dh[:,i,:]+dh0, cache.pop())
        dWx+=dWx_
        dWh+=dWh_
        db+=db_
   
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  (N, T)=x.shape
  (V, D)=W.shape
  out=np.zeros(( (N, T, D) ))
  inds=[]
  for n in range(N):   
      for t in range(T):
            idx=x[n,t]
            out[n,t]=W[idx]
            inds.append(idx)
  cache=(x,W,inds)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  (x,W,inds)=cache
  dW = np.zeros(W.shape)
  (N, T)=x.shape
  inds=[]
  for n in range(N):   
      for t in range(T):
            idx=x[n,t]
            dW[idx]+=dout[n,t,:]
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW



def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  N,H=prev_h.shape
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  term_input=np.dot(x,Wx)+b #R^{N,H}, bias is ony for the input weights
  term_prevstate=np.dot(prev_h,Wh) #R^{N,H}
  input=term_prevstate+term_input
  a1=input[:,:H]
  a2=input[:,H:2*H]
  a3=input[:,2*H:3*H]
  a4=input[:,3*H:]
  i=sigmoid(a1)
  f=sigmoid(a2)
  o=sigmoid(a3)
  g=np.tanh (a4) #doing an acivation on the sum of inputs
   
  next_c=f*prev_c+i*g
  next_h=o*np.tanh(next_c)
    
  cache=(x, prev_h, prev_c, Wx, Wh, b,i,f,o,g,a1,a2,a3,a4,next_c) #??????
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  
  
  
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
   
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  (N,H)=dnext_h.shape
  (x, prev_h, prev_c, Wx, Wh, b,i,f,o,g,a1,a2,a3,a4,next_c)=cache
  dnext_c_new=dnext_c+tanh_dx(next_c)*dnext_h*o
    
  da1=dnext_c_new*g*sig_dx(a1) #??
  da2=dnext_c_new*prev_c*sig_dx(a2)#??
    
  da3=dnext_h*np.tanh(next_c)*sig_dx(a3)#????????

  da4=dnext_c_new*i*tanh_dx(a4)
  dx, dprev_h, dprev_c, dWx, dWh, db = np.zeros(x.shape), np.zeros(prev_h.shape),np.zeros(prev_c.shape), np.zeros(Wx.shape), np.zeros(Wh.shape), np.zeros(b.shape)

  db[:H]=np.sum(da1,axis=0)
  db[H:2*H]=np.sum(da2,axis=0)
  db[2*H:3*H]=np.sum(da3,axis=0)
  db[3*H:]=np.sum(da4,axis=0)
  
 
  dprev_c=dnext_c_new*f

  
  dprev_h=np.dot(da1,Wh[:,:H].T) #N,H
  dprev_h+=np.dot(da2,Wh[:,H:2*H].T)
  dprev_h+=np.dot(da3,Wh[:,2*H:3*H].T)
  dprev_h+=np.dot(da4,Wh[:,3*H:].T)
  

  dWh[:,:H]=np.dot(prev_h.T,da1)
  dWh[:,H:2*H]=np.dot(prev_h.T,da2)
  dWh[:,2*H:3*H]=np.dot(prev_h.T,da3)
  dWh[:,3*H:]=np.dot(prev_h.T,da4)
    
  dWx[:,:H]=np.dot(x.T,da1)
  dWx[:,H:2*H]=np.dot(x.T,da2)
  dWx[:,2*H:3*H]=np.dot(x.T,da3)
  dWx[:,3*H:]=np.dot(x.T,da4)
    
  dx=np.dot(da1,Wx[:,:H].T)+np.dot(da2,Wx[:,H:2*H].T)+np.dot(da3,Wx[:,2*H:3*H].T)+np.dot(da4,Wx[:,3*H:].T) #N,D
 
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and 
    
  #you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
   
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, []
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################=h0
  prev_h=h0
  (N, T, D)=x.shape
  (N, H)=prev_h.shape
  prev_c=np.zeros((N,H))
  h=np.zeros((N, T, H))
  for i in range(T):
        prev_h, prev_c, C=lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        h[:,i,:]=prev_h
        cache.append(C)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  #dx, dh0, dWx, dWh, db = None, None, None, None, None
   
  
  #init
  D=cache[0][0].shape[1]
  (N, T, H)=dh.shape
  dx, dprev_h, dWx, dWh, db =None,  np.zeros((N,H)),  np.zeros((D,4*H)),  np.zeros((H,4*H)),  np.zeros((4*H,)) 
  dx=np.zeros((N, T, D))
  dprev_c= np.zeros((N,H)) 
  dx[:,T-1,:], dprev_h, dprev_c, dWx, dWh, db=lstm_step_backward(dh[:,T-1,:],dprev_c, cache.pop())
  for i in reversed(range(0,T-1)):
        dx[:,i,:], dprev_h, dprev_c, dWx_, dWh_, db_=lstm_step_backward(dh[:,i,:]+dprev_h,dprev_c, cache.pop())
        dWx+=dWx_
        dWh+=dWh_
        db+=db_
        
        
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dprev_h, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

