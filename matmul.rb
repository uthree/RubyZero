require "./lib/rubyzero.rb"
include RubyZero

a = FloatTensor[
    [1, 2],
    [3, 4],
    [5, 6]
]
a.requires_grad = true

b = FloatTensor[
    [1, 2, 3],
    [4, 5, 6]
]
b.requires_grad = true

100. times do
    c = a.dot(b)
    c = c.sum.sum
    c.backward
    a -= a.grad_tensor * 0.01
end
p a
