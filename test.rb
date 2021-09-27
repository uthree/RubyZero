require_relative './lib/rubyzero.rb'

a = RubyZero::Float64[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

b = RubyZero::Float64[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

c = a * b
c.backward
p a.grad