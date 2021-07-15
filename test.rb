require "./lib/ruby_zero.rb"
include RubyZero
a = Tensor[1,2,3]
b = Tensor[4,5,6]
c = Tensor[7,8,9]
output = a * b * c
output.backward
p c.grad