import numpy as np
from mnist import training_data, print_img
import ml_utils

def main():
  imgs, labels = training_data()
  X = np.array(imgs, dtype=float)
  X = (X/256.0).reshape(imgs.shape[0], imgs.shape[1]*imgs.shape[2])+1
  y_real = np.zeros((labels.size, labels.max()+1))
  y_real[np.arange(labels.size), labels] = 1
  X_t = np.transpose(X)
  w = np.random.rand(X.shape[1], labels.max()+1)*2-1
  for t in range(500):
    res = np.matmul(X, w)-y_real
    grad = 2*np.matmul(X_t, res)
    w-=.00000001*grad
    y_model = np.matmul(X, w)
    #svm
    c = ml_utils.Constant(y_model)
    svm = ml_utils.SVMLoss(c)
    loss = svm.eval({'labels':labels}, {})
    print('svm: {}'.format(loss))
    #svm
    err = np.sum((y_real-y_model)**2)
    print('{}: {}'.format(t, err))
    if t % 25 == 0:
      hits = 0
      for i in range(X.shape[0]):
        x = X[i,:]
        y = np.matmul(x, w)
        predicted = np.argmax(y)
        actual = labels[i]
        if predicted == actual:
          hits += 1
      print('hit rate: {}'.format(hits/X.shape[0]))  

if __name__ == '__main__':
  main()
