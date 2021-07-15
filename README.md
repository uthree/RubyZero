# RubyZero
A simple deep learning library for Ruby.
This library is likes [PyTorch](https://github.com/pytorch/pytorch) and [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).

# Example
```ruby
require "./lib/ruby_zero.rb"
include RubyZero

a = Tensor[1,2,3]
b = Tensor[4,5,6]
c = Tensor[7,8,9]
output = a * b * c
output.backward
p c.grad
```

# Installation
comming soon.

# Documentation
comming soon.