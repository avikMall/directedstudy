import tvm
import numpy as np
from tvm import te

# tgt = tvm.target.Target(target="llvm", host="llvm")

m = te.var("m") # define variable
n = te.var("n") # define variable
A = te.placeholder((m, n), name="A") 
B = te.placeholder((m, n), name="B") 
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")
s = te.create_schedule([C.op])

print(tvm.lower(s, [A, B, C], simple_mode=True))




# mod = tvm.build(s, [A, B, C], target = "llvm")



# X = np.array([[4,3], [13, 7]])
# Y = np.array([[2, 2], [2, 2]])
# Z = np.array([[0,0], [0,0]])






# A = tvm.nd.array(X.astype("float32"))
# B = tvm.nd.array(Y.astype("float32"))
# C = tvm.nd.array(Z.astype("float32"))

# mod(A, B, C)
# print(C)


