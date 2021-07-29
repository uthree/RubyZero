require_relative "./core.rb"


module RubyZero::Optimizers
    class SGD < Optimizer
        def initialize(learning_rate:0.01)
            @lr = learning_rate
            super
        end
        def update_parameter(tensor)
            tensor -= tensor.grad * @lr
            return tensor
        end
    end
end