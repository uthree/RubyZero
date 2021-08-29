module RubyZero::NN::Optimizers
    class SGD < Optimizer
        def initialize(parameters, lr: 0.01)
            super(parameters, lr)
        end
        def step
            @parameters.each do |param|
                param.value -= param.value.grad_tensor * @learning_rate
            end
        end
    end
end