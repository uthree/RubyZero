require "./lib/rubyzero.rb"
include RubyZero
a = Tensor[1,2,3]
b = Tensor[1,2,3]
c = a + b
d = Tensor[
    [1,2,3],
    [4,5,6],
]
p c