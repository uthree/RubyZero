

module RubyZero
    module Optimizers

    end
end

module RubyZero::Optimizers
    class Optimizer
        public def initialize(*args, **kwargs, &block)
            @parameters = RubyZero::NN::Parameters.new()
        end
        # add module parameters
        public def <<(nn_module)
            @parameters = @parameters + nn_module.parameters
            on_add_module(nn_module)
        end
        public def step()
            @parameters.elements.each do |tensor|
                tensor.data = update_parameter(tensor).data
            end
        end
        public def init_gradients()
            @parameters.elements.each do |tensor|
                tensor.init_gradients()
            end
        end
        alias_method :zero_grad, :init_gradients
        alias_method :init_grad, :init_gradients
        private def update_parameter(tensor)

        end

        private def on_add_module(nn_module)
        end
    end
end