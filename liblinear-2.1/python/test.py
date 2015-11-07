from liblinearutil import *

y, x = svm_read_problem('../heart_scale')
m = train(y[:200], x[:200], '-s 2 -C')
p_label, p_acc, p_val = predict(y[200:], x[200:], m)
print p_acc
