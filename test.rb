require "./lib/ruby_zero.rb"
include RubyZero

a = Tensor[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

p a.repeat(5, axis:1)