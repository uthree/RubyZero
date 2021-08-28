require "./lib/rubyzero.rb"
include RubyZero
a = Tensor[1,2,3,4,5,6,7,8,9,10]
a = a.repeat(3, axis:0)
p a
