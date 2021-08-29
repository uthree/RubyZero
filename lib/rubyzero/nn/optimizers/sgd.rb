module RubyZero::NN::Optimizers
    class SGD < Optimizer
        def initialize(parameters, lr: 0.01)
            super(parameters, lr)
            @learning_rate = lr
        end
        def step
            p "OPTIM STEP"
            @parameters.each_with_index do |param, index|
                p index
                p param.value.grad_tensor
                #param.value -= param.value.grad_tensor * @learning_rate
            end
        end
    end
end