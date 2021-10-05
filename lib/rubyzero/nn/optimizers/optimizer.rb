module RubyZero::NN::Optimizers
    # base of optimizer classes
    class Optimizer
        def initialize(parameters, lr:0.01)

        end
        def update()

        end
        def zero_grad()
            @parameters.each do |parameter|
                parameter.grad = parameter.zeros_like()
            end
        end
    end
end