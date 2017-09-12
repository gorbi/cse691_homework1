import numpy as np
import time
import statistics as st

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
z_1 = [[0 for j in range(5)] for i in range(3)]
# NumPy
z_2 = np.zeros((3, 5))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
z_1[0] = [7 for j in z_1[0]]
# NumPy
z_2[0, :] = 7

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
for i in range(len(z_1)):
    z_1[i][1] = 9
# NumPy
z_2[:, 1] = 9

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
z_1[1][2] = 5
# NumPy
z_2[1, 2] = 5

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
x_1 = [x+50 for x in range(50)]
# NumPy
#x_2 = np.arange(50, 100)
x_2 = np.linspace(50, 99, 50)

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
y_1 = [[j+i*4 for j in range(4)] for i in range(4)]
# NumPy
y_2 = np.arange(16).reshape(4, 4)

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
tmp_1 = [[1 for j in range(5)] for i in range(5)]
for i in range(len(tmp_1)-2):
    for j in range(len(tmp_1)-2):
        tmp_1[i+1][j+1] = 0
# NumPy
tmp_2 = np.ones(25).reshape(5, 5)
tmp_2[1:4, 1:4] = 0

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
a_1 = [[int(j+i*100) for j in range(100)] for i in range(50)]
# NumPy
a_2 = np.arange(0, 5000, dtype='int').reshape(50, 100)

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
b_1 = [[int(j+i*200) for j in range(200)] for i in range(100)]
# NumPy
b_2 = np.arange(0, 20000, dtype='int').reshape(100, 200)

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
c_1 = [[0 for j in range(len(b_1[0]))] for i in range(len(a_1))]
for i in range(len(a_1)):
    for j in range(len(b_1[0])):
        for k in range(len(a_1[0])):
            c_1[i][j] += a_1[i][k]*b_1[k][j]
# NumPy
c_2 = a_2.dot(b_2)

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python

# NumPy


##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
a_1 = [[a_1[i][j]-st.mean(a_1[i]) for j in range(len(a_1[0]))] for i in range(len(a_1))]
# NumPy
a_2 = a_2 - a_2.mean(axis=1, keepdims=True)

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
b_1 = [[b_1[i][j]-st.mean([b_1[k][j] for k in range(len(b_1))]) for j in range(len(b_1[0]))] for i in range(len(b_1))]
# NumPy
b_2 = b_2 - b_2.mean(axis=0, keepdims=True)

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
e_1 = map(list, zip(*c_1))
e_1 = [i for i in e_1]
e_1 = [[e_1[i][j]+5 for j in range(len(e_1[0]))] for i in range(len(e_1))]
# NumPy
e_2 = c_2.T+5

##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
f_1 = e_1[0]
for i in range(len(e_1)-1):
    f_1.extend(e_1[i+1])
print(np.array(f_1).shape)
# NumPy
f_2 = np.reshape(e_2, (e_2.size,))
print(f_2.shape)
