import numpy as np
import time
import statistics as st
import random

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime = time.time()
z_1 = [[0 for j in range(5)] for i in range(3)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3, 5))
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
pythonStartTime = time.time()
z_1[0] = [7 for j in z_1[0]]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[0, :] = 7
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
pythonStartTime = time.time()
for i in range(len(z_1)):
    z_1[i][1] = 9
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[:, 1] = 9
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
z_2[1, 2] = 5
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime = time.time()
x_1 = [x+50 for x in range(50)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
#x_2 = np.arange(50, 100)
x_2 = np.linspace(50, 99, 50)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
pythonStartTime = time.time()
y_1 = [[j+i*4 for j in range(4)] for i in range(4)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
y_2 = np.arange(16).reshape(4, 4)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [[1 for j in range(5)] for i in range(5)]
for i in range(len(tmp_1)-2):
    for j in range(len(tmp_1)-2):
        tmp_1[i+1][j+1] = 0
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones(25).reshape(5, 5)
tmp_2[1:4, 1:4] = 0
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
pythonStartTime = time.time()
a_1 = [[int(j+i*100) for j in range(100)] for i in range(50)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
a_2 = np.arange(0, 5000, dtype='int').reshape(50, 100)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
b_1 = [[int(j+i*200) for j in range(200)] for i in range(100)]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
b_2 = np.arange(0, 20000, dtype='int').reshape(100, 200)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
pythonStartTime = time.time()
c_1 = [[0 for j in range(len(b_1[0]))] for i in range(len(a_1))]
for i in range(len(a_1)):
    for j in range(len(b_1[0])):
        for k in range(len(a_1[0])):
            c_1[i][j] += a_1[i][k]*b_1[k][j]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
c_2 = a_2.dot(b_2)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
pythonStartTime = time.time()
d_1 = [[random.randint(0, 1337) for j in range(3)] for i in range(3)]
d_1_1d = d_1[0].copy()
for i in range(len(d_1)-1):
    d_1_1d.extend(d_1[i+1])
d_1_min_maxmin = [min(d_1_1d)]
d_1_min_maxmin.append(max(d_1_1d)-d_1_min_maxmin[0])
d_1 = [[(d_1[i][j]-d_1_min_maxmin[0])/(d_1_min_maxmin[1]) for j in range(len(d_1[0]))] for i in range(len(d_1))]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
d_2 = np.random.randint(0, 1337, (3, 3))
d_2 = (d_2 - d_2.min()) / (d_2.max()-d_2.min())
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
pythonStartTime = time.time()
a_1 = [[a_1[i][j]-st.mean(a_1[i]) for j in range(len(a_1[0]))] for i in range(len(a_1))]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(axis=1, keepdims=True)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
pythonStartTime = time.time()
b_1 = [[b_1[i][j]-st.mean([b_1[k][j] for k in range(len(b_1))]) for j in range(len(b_1[0]))] for i in range(len(b_1))]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis=0, keepdims=True)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
pythonStartTime = time.time()
e_1 = map(list, zip(*c_1))
e_1 = [i for i in e_1]
e_1 = [[e_1[i][j]+5 for j in range(len(e_1[0]))] for i in range(len(e_1))]
pythonEndTime = time.time()
# NumPy
numPyStartTime = time.time()
e_2 = c_2.T+5
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
pythonStartTime = time.time()
f_1 = e_1[0]
for i in range(len(e_1)-1):
    f_1.extend(e_1[i+1])
pythonEndTime = time.time()
print(np.array(f_1).shape)
# NumPy
numPyStartTime = time.time()
f_2 = np.reshape(e_2, (e_2.size,))
numPyEndTime = time.time()
print(f_2.shape)
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
