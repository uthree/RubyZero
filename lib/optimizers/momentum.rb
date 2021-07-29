require_relative "./core.rb"

module RubyZero::Optimizers
    class Momentum < Optimizer
        def initialize(learning_rate: 0.1, momentum: 0.9)
            super()
            @lr = learning_rate
            @velocity = {}
            @momentum = momentum
        end

        def on_add_module(nn_module)
            nn_module.parameters.each { |param|
                @velocity[param.id] = param.zeros_like
            }
        end

        def update_parameter(tensor)
            @velocity[tensor.id] = @velocity[tensor.id] * @momentum + tensor.grad * @lr
            tensor -= @velocity[tensor.id]
        end
    end
end