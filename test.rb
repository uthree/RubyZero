require "./lib/rubyzero.rb"
include RubyZero
a = Tensor[1,2,3]
b = Tensor[3,4,5]
c = a * b
c.backward
p a.grad_tensor