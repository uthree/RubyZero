require_relative './lib/rubyzero.rb'

affine = RubyZero::NN::Layers::Affine.new(2, 3)
input = RubyZero::Float64[
    [1,2],
    [3,4],
]
output = affine.call(input)
p affine
