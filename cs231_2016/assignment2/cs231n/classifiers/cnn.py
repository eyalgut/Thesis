import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
        
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
    
    self.params['W2']=weight_scale*np.random.randn(num_filters*HO*WO,hidden_dim)
    self.params['b2']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['b3']=np.zeros((num_classes,1))                          
    self.params['W3']=weight_scale*np.random.randn(hidden_dim,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    out,cache_aff_relu=affine_relu_forward(out, W2, b2)
    
    scores, aff_cache = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw3,db3 = affine_backward(d_scores, aff_cache  )
    dout, dw2, db2 = affine_relu_backward(dout, cache_aff_relu)
    dx, dw1, db1=conv_relu_pool_backward(dout, cache_conv_pool)
    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

class ThreeLayerConvNet2(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
        
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
    
    
    input_dim=(num_filters,HO,WO)
    self.params['b2']=np.zeros((num_filters,1)).ravel()
    self.params['W2']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
        
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
    
    input_dim=(num_filters,HO,WO)
    self.params['b3']=np.zeros((num_classes,1))                          
    self.params['W3']=weight_scale*np.random.randn(np.prod(input_dim),num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #out,cache_aff_relu=affine_relu_forward(out, W2, b2)
    out,cache_conv_pool_2=conv_relu_pool_forward(out, W2, b2, conv_param, pool_param)
    
    scores, aff_cache = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw3,db3 = affine_backward(d_scores, aff_cache  )
    dout, dw2, db2=conv_relu_pool_backward(dout, cache_conv_pool_2)
    dx, dw1, db1=conv_relu_pool_backward(dout, cache_conv_pool)
    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

class ThreeLayerConvNet3(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
        
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
    
    
    input_dim=(num_filters,HO,WO)
    self.params['b2']=np.zeros((num_filters,1)).ravel()
    self.params['W2']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
   

    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
 
    input_dim=(num_filters,HO,WO)
    
    
    self.params['W3']=weight_scale*np.random.randn(num_filters*HO*WO,hidden_dim)
    self.params['b3']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['b4']=np.zeros((num_classes,1))                          
    self.params['W4']=weight_scale*np.random.randn(hidden_dim,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #out,cache_aff_relu=affine_relu_forward(out, W2, b2)
    out,cache_conv_pool_2=conv_relu_pool_forward(out, W2, b2, conv_param, pool_param)
    
    
    out,cache_aff_relu=affine_relu_forward(out, W3, b3)
    
    scores, aff_cache = affine_forward(out, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw4,db4 = affine_backward(d_scores, aff_cache  )
    
    dout, dw3, db3 = affine_relu_backward(dout, cache_aff_relu)
        
    dout, dw2, db2=conv_relu_pool_backward(dout, cache_conv_pool_2)
    dx, dw1, db1=conv_relu_pool_backward(dout, cache_conv_pool)
    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    dw4+=reg*params['W4']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    grads['W4']=dw4
    grads['b4']=db4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

class ThreeLayerConvNet4(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    print '4444'
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
        
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride
    
    
    input_dim=(num_filters,HO,WO)
    self.params['b2']=np.zeros((num_filters,1)).ravel()
    self.params['W2']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    HO = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    WO = 1 + (input_dim[2]+2*pad  - filter_size) / stride
   

 
    input_dim=(num_filters,HO,WO)
    
    
    self.params['W3']=weight_scale*np.random.randn(num_filters*HO*WO,hidden_dim)
    self.params['b3']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['W4']=weight_scale*np.random.randn(hidden_dim,hidden_dim)
    self.params['b4']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['b5']=np.zeros((num_classes,1))                          
    self.params['W5']=weight_scale*np.random.randn(hidden_dim,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #out,cache_aff_relu=affine_relu_forward(out, W2, b2)
    out,cache_conv_pool_2=conv_relu_forward(out, W2, b2, conv_param)
 
    
    out,cache_aff_relu=affine_relu_forward(out, W3, b3)
    
    out,cache_aff_relu_2=affine_relu_forward(out, W4, b4)
        
    scores, aff_cache = affine_forward(out, W5, b5)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw5,db5 = affine_backward(d_scores, aff_cache  )
    dout, dw4, db4 = affine_relu_backward(dout, cache_aff_relu_2)
    dout, dw3, db3 = affine_relu_backward(dout, cache_aff_relu)
        
    dout, dw2, db2=conv_relu_backward(dout, cache_conv_pool_2)
    dx, dw1, db1=conv_relu_pool_backward(dout, cache_conv_pool)
    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    dw4+=reg*params['W4']
    dw5+=reg*params['W5']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    grads['W4']=dw4
    grads['b4']=db4
    grads['W5']=dw5
    grads['b5']=db5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
      
class ThreeLayerConvNet5(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    print '5555'
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
    
    input_dim=(num_filters,H_,W_)
    
    self.params['b2']=np.zeros((num_filters,1)).ravel()
    self.params['W2']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    

    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride

 
    input_dim=(num_filters,HO,WO)
    
    
    self.params['W3']=weight_scale*np.random.randn(num_filters*HO*WO,hidden_dim)
    self.params['b3']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['W4']=weight_scale*np.random.randn(hidden_dim,hidden_dim)
    self.params['b4']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    #self.params['W6']=weight_scale*np.random.randn(hidden_dim,hidden_dim)
   # self.params['b6']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['b5']=np.zeros((num_classes,1))                          
    self.params['W5']=weight_scale*np.random.randn(hidden_dim,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
   # W6, b6 = self.params['W6'], self.params['b6']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_forward(X, W1, b1, conv_param)
    
    #out,cache_aff_relu=affine_relu_forward(out, W2, b2)
    out,cache_conv_pool_2=conv_relu_pool_forward(out, W2, b2, conv_param,pool_param,)
 
    
    out,cache_aff_relu=affine_relu_forward(out, W3, b3)
    
    out,cache_aff_relu_2=affine_relu_forward(out, W4, b4)
    
   # out,cache_aff_relu_6=affine_relu_forward(out, W6, b6)
    
    scores, aff_cache = affine_forward(out, W5, b5)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw5,db5 = affine_backward(d_scores, aff_cache  )
   ## dout, dw6, db6 = affine_relu_backward(dout, cache_aff_relu_6)
    dout, dw4, db4 = affine_relu_backward(dout, cache_aff_relu_2)
    dout, dw3, db3 = affine_relu_backward(dout, cache_aff_relu)
        
    dout, dw2, db2=conv_relu_pool_backward(dout, cache_conv_pool_2)
 
    dx, dw1, db1=conv_relu_backward(dout, cache_conv_pool)
    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    dw4+=reg*params['W4']
    dw5+=reg*params['W5']
  #  dw6+=reg*params['W6']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    grads['W4']=dw4
    grads['b4']=db4
    grads['W5']=dw5
    grads['b5']=db5
   # grads['W6']=dw5
   # grads['b6']=db5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
class ThreeLayerConvNet6(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  Possibly change arch to: Should help get that 65% acc
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax
  Only need to change INIT,FORWARD,BACKWARD PASS !!!
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    print '666'
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params['b1']=np.zeros((num_filters,1)).ravel()
    self.params['W1']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
  
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
    
    input_dim=(num_filters,H_,W_)
    
    self.params['b2']=np.zeros((num_filters,1)).ravel()
    self.params['W2']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    
    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
    
    input_dim=(num_filters,H_,W_)
    
    self.params['b3']=np.zeros((num_filters,1)).ravel()
    self.params['W3']=weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size) 
    

    
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    stride=conv_param['stride']
    pad=conv_param['pad']
    H_ = 1 + (input_dim[1]+2*pad  - filter_size) / stride
    W_ = 1 + (input_dim[2]+2*pad  - filter_size) / stride
    pad=0
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    HO = 1 + (H_ + 2 * pad - pool_height) / stride
    WO = 1 + (W_ + 2 * pad - pool_width) / stride

 
    input_dim=(num_filters,HO,WO)
    
    
    self.params['W4']=weight_scale*np.random.randn(num_filters*HO*WO,hidden_dim)
    self.params['b4']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
    
    self.params['W5']=weight_scale*np.random.randn(hidden_dim,hidden_dim)
    self.params['b5']=np.zeros((hidden_dim,1))#possibly change to dim (H,1)
 
    self.params['b6']=np.zeros((num_classes,1))                          
    self.params['W6']=weight_scale*np.random.randn(hidden_dim,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache_conv_pool=conv_relu_forward(X, W1, b1, conv_param)
    
    out,cache_conv_pool_3=conv_relu_forward(out, W2, b2, conv_param)
    out,cache_conv_pool_2=conv_relu_pool_forward(out, W3, b3, conv_param,pool_param)
 
    out,cache_aff_relu=affine_relu_forward(out, W4, b4)
    
    out,cache_aff_relu_2=affine_relu_forward(out, W5, b5)
   
    
    scores, aff_cache = affine_forward(out, W6, b6)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    reg=self.reg
    params=self.params
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)#forward soft max
    
    loss+=0.5*reg*(np.linalg.norm(params['W1'])**2+np.linalg.norm(params['W2'])**2+np.linalg.norm(params['W3'])**2) #no regulization on the biases :)
      
    dout,dw6,db6 = affine_backward(d_scores, aff_cache  )
    dout, dw5, db5 = affine_relu_backward(dout, cache_aff_relu_2)
    dout, dw4, db4 = affine_relu_backward(dout, cache_aff_relu)
        
    dout, dw3, db3=conv_relu_pool_backward(dout, cache_conv_pool_2)
 
    dout, dw2, db2=conv_relu_backward(dout, cache_conv_pool_3)
    dx, dw1, db1=conv_relu_backward(dout, cache_conv_pool)
    

    ############################################################################
    #do backward pass
    #                             END OF YOUR CODE                             #
    ############################################################################
    dw1+=reg*params['W1']
    dw2+=reg*params['W2']
    dw3+=reg*params['W3']
    dw4+=reg*params['W4']
    dw5+=reg*params['W5']
    dw6+=reg*params['W6']
    grads['W1']=dw1
    grads['W2']=dw2
    grads['W3']=dw3
    grads['b1']=db1
    grads['b2']=db2
    grads['b3']=db3
    grads['W4']=dw4
    grads['b4']=db4
    grads['W5']=dw5
    grads['b5']=db5
    grads['W6']=dw5
    grads['b6']=db5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
            
pass

