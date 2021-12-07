module RubyZero::NN::Optimizers
    class SGD < Optimizer
        def initialize(parameters, lr: 0.01)
            @lr = lr
            @parameters = parameters
        end
        def update()
            @parameters.each do |tensor|
                tensor.data -= tensor.grad.data * @lr
            end
        end
        alias :step :update
    end
end