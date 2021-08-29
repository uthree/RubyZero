module RubyZero::NN::Layers

end

require_relative './layer.rb'
require_relative './linear.rb'
require_relative './affine.rb'
require_relative './relu.rb'
require_relative './sigmoid.rb'
require_relative './softmax.rb'


module RubyZero
    L = NN::Layers
end