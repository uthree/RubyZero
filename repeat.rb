require "./lib/rubyzero.rb"
include RubyZero

tensor = Tensor[2,3,4]
tensor.requires_grad = true
o = tensor.sum
o.backward
p o
