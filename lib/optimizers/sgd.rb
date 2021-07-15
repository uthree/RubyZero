module RubyZero::Optimizers
    class SGD < Optimizer
        def initialize(learning_rate=0.01)
            @lr = learning_rate
            super
        end
        def step()
            @parameters.elements.each do |tensor|
                tensor -= @lr * tensor.grad
            end
        end
    end
end