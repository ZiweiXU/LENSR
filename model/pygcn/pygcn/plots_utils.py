import json
import matplotlib.pyplot as plt
import numpy as np

train_save_syntax = json.load(open('boonsyntax.train_save'))
train_save_ddnnf = json.load(open('boonsimple.train_save'))
test_save_syntax = json.load(open('boonsyntax.test_save'))
test_save_ddnnf = json.load(open('boonsimple.test_save'))

fig1 = plt.figure()
x = np.arange(len(train_save_ddnnf)-1)
plt.plot(x,train_save_ddnnf[1:],label='ddnnf')
plt.plot(x,train_save_syntax[1:],label='syntax')
plt.legend()
plt.savefig('train_plot')

fig2 = plt.figure()
x = np.arange(len(test_save_ddnnf)-1)
plt.plot(x,test_save_ddnnf[1:],label='ddnnf')
plt.plot(x,test_save_syntax[1:],label='syntax')
plt.legend()
plt.savefig('test_plot')