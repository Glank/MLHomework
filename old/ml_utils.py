import numpy as np

class Differentiable:
  def shape(self):
    raise NotImplementedError()
  def eval(self, inputs, weights):
    raise NotImplementedError()
  def diff(self, inputs, weights):
    raise NotImplementedError()

class Constant(Differentiable):
  def __init__(self, value):
    self.value = value
  def shape(self):
    return self.value.shape
  def eval(self, inputs, weights):
    return self.value
  def diff(self, inputs, weights):
    return {}

class WeightArray(Differentiable):
  def __init__(self, name, shape):
    self.name = name
    self.shape = shape
  def shape(self):
    return self.shape
  def eval(self, inputs, weights):
    return weights[self.name]
  def diff(self, inputs, weights):
    grad = np.zeros(self.shape+self.shape)
    for i in range(weights[self.name].size):
      w_idx = np.unravel_index(i, self.shape)
      grad[w_idx+w_idx] = 1
    gradient = {}
    gradient[self.name] = grad
    return gradient

class SVMLoss(Differentiable):
  def __init__(self, in_func, labels_key='labels'):
    self.in_func = in_func
    self.labels_key = 'labels'
  def shape(self):
    return tuple()
  def eval(self, inputs, weights):
    scores = self.in_func.eval(inputs, weights)
    labels = inputs[self.labels_key]
    assert scores.shape[0] == labels.shape[0]
    desired_scores = scores[np.arange(labels.size),labels].reshape((labels.size,1))
    return np.array([np.sum(np.maximum(0, 1+scores-desired_scores))-desired_scores.size])
  def diff(self, inputs, weights):
    # ds(f(w))/dw = ds/df * df/dw
    dfdw = self.in_func.diff(inputs, weights)
    labels = inputs[self.labels_key]
    scores = self.in_func.eval(inputs, weights)
    desired_scores = scores[np.arange(labels.size),labels].reshape((labels.size,1))
    lines = 1+scores-desired_scores
    #ds/df[i,l] =
    #  if i!=d: 0 if 1+s[i,l]-s_[i,d] < 0 else 1
    #  if i==d: 1-(num lines>0)
    dsdf = np.ones((1,)+scores.shape)
    dsdf[0,np.argwhere(lines<0)] = 0
    dsdf[0,np.arange(labels.size),labels] = 1-np.sum(dsdf, axis=2)
    gradient = {}
    for name, dfdwi in dfdw.items():
      gradient[name] = np.tensordot(dsdf, dfdwi, 2)
    return gradient

def test_gradient(differentiable, inputs, weights):
  epsilon = 0.00001
  err = 0
  act_gradient = differentiable.diff(inputs, weights)
  init = np.copy(differentiable.eval(inputs, weights))
  for name, weight in weights.items():
    for i in range(weight.size):
      w_idx = np.unravel_index(i, weight.shape)
      weight[w_idx] += epsilon
      end = np.copy(differentiable.eval(inputs, weights))
      weight[w_idx] -= epsilon
      for j in range(end.size):
        d_idx = np.unravel_index(j, end.shape) 
        partial = (end[d_idx]-init[d_idx])/epsilon
        individual_err = abs(partial-act_gradient[name][d_idx+w_idx])
        if individual_err > 0.1:
          print('{} error at {}'.format(individual_err, (d_idx+w_idx)))
          print('est: {}'.format(partial))
          print('act: {}'.format(act_gradient[name][d_idx+w_idx]))
        err += individual_err
  return err

def main():
  labels = np.array([1, 0, 1])
  w = np.array([
    [0.1, 0.9],
    [0, 2],
    [0, 0.1]
  ])
  diffable_w = WeightArray('w', w.shape)
  model = SVMLoss(diffable_w)
  weights = {'w': w}
  inputs = {'labels':labels}
  print(test_gradient(model, inputs, weights))
  for t in range(100):
    loss = model.eval(inputs, weights)
    print('{}) loss: {}'.format(t, loss))
    gradient = model.diff(inputs, weights)
    for name in weights:
      weights[name] -= 0.01*gradient[name][0,]

if __name__=='__main__':
  main()
