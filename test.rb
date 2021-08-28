require "./lib/rubyzero.rb"
include RubyZero
a = FloatTensor[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
a = F.softmax(a)
p a