from svmutil import *


y, x = svm_read_problem('test')
m = svm_train(y[:2], x[:2], '-c 4')
p_label, p_acc, p_val = svm_predict(y[2:], x[2:], m)
accuracy = p_acc[0]

print "accuracy: ", accuracy
print "p_label: ", p_label
print "p_acc: ", p_acc
print "p_val: ", p_val

y, x = svm_read_problem('heart_scale')
y_test, x_test = svm_read_problem('heart_scale_test')
m = svm_train(y[:100], x[:100], '-c 4')
p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
accuracy = p_acc[0]

print "accuracy: ", accuracy
print "p_label: ", p_label
print "p_acc: ", p_acc
print "p_val: ", p_val