module RubyZero::NN::Optimizers
    class Optimizer
        def initialize(parameters, learning_rate)
            @parameters = parameters
            @learning_rate = learning_rate
        end
        def zero_grad
            @parameters.each do |param|
                param.value.init_grad_tensor
            end
        end
        def step
        end
    end
end