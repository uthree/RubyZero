require "./lib/rubyzero.rb"
include RubyZero
a = Tensor[1,2,3]
b = Tensor[1,2,3]
c = a + b
p c