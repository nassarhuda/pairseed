using SparseArrays
using MatrixNetworks
using LinearAlgebra
function triproducts(A,x,y)
  z = similar(x)
  z .= 0
  T = triangles(A;symmetries=true)
  for tri in T
    z[tri.v1] += x[tri.v2]*y[tri.v3]
  end
  return z
end
function trpr(A, x0::Vector{Float64}, α, iter)
  # assumes A is a symmetric sparse matrix
  # T is a list of triangles,and fully symmetric.
  @assert(issymmetric(A))
  x = copy(x0)
  n = size(A,1)
  dg = vec(sum(A;dims=1))
  for i=1:iter
    dx = triproducts(A,x,ones(n))
    d = dx + dg
    id = map!(x -> 1.0/x, d, d)
    y = x.*id
    xn = α.*triproducts(A,x,y) + α.*(A*y) + (1-α).*x0
    x = xn 
  end
  return x
end
