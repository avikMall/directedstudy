import featgraph, tvm
A = featgraph.spmat(shape=(n,n), nnz=m)

XV = tvm.placeholder(shape=(n,d))
def msgfunc(src, dst, eid):
    out = tvm.compute((d,), lambda i: XV[src,i])
    return out

# tile feature dimension for cache optimization ---- from FeatGraph Paper
def cpu_schedule(out):
    s = tvm.create_schedule(out)
    s[out].split(out.axis[0], factor=8)
    return s

# parallelize feature dimension by using CUDA
def gpu_schedule(out):
    s = tvm.create_schedule(out)
    s[out].bind(out.axis[0], 'thread.x')
    return s

# use sum as the aggregation function
aggregation = tvm.sum

# run spmm
if target = 'cpu'':
    fds = cpu_schedule
elif target == 'gpu':
    fds = gpu_schedule
    GCN = featgraph.spmm(A, msgfunc, aggregation, target, fds)
