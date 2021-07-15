module RubyZero::Optimizers
    class SGD < Optimizer
        def initialize(learning_rate=0.01)
            @lr = learning_rate
            super
        end
        def step()
            @parameters.elements.map do |tensor|
                #p "a"
                tensor.data = tensor.data - @lr * tensor.grad.data
            end
        end
    end
end