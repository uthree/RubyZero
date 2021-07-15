

module RubyZero
    module Optimizers

    end
end

module RubyZero::Optimizers
    class Optimizer
        def initialize(*args, **kwargs, &block)
            @parameters = RubyZero::NN::Parameters.new()
        end
        # add module parameters
        def <<(nn_module)
            @parameters = @parameters + nn_module.parameters
        end
        def step()

        end
        def init_gradients()
            @parameters.elements.each do |tensor|
                tensor.init_gradients()
            end
        end
        alias_method zero_grad, :init_gradients
        alias_method init_grad, :init_gradients
    end
end