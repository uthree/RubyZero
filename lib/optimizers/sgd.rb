module RubyZero::Optimizers
    class SGD < Optimizer
        def initialize(learning_rate=0.01)
            @lr = learning_rate
            super
        end
        def update_parameter(tensor)
            tensor.data -= @lr * tensor.grad.data
            return tensor
        end
    end
end