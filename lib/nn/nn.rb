module RubyZero
    module NN

    end
end

module RubyZero::NN
    class Module # template module of neural network
        attr_reader :__parameters__, :__childlen__
        def initialize()
            @__parameters__ = Parameters.new
            @__childlen__ = []
            @__flag_init_update__ = false
        end

        def call(*args, **kwargs, &block)
            forward(*args, **kwargs, &block)
        end

        def forward(*args, **kwargs, &block)
            raise NotImplementedError, "#{self.class}.forward method not implemented"
        end

        def eval()
            @__parameters__.eval
        end
        def train()
            @__parameters__.train
        end

        def __update_childlen__()
            inv = instance_variables
            inv.delete(:@__parameters__)
            inv.delete(:@__childlen__)
            inv.delete(:@__flag_init_update__)

            inv.each do |k|
                value = instance_variable_get(k)
                if value.is_a?(RubyZero::Tensor)
                    @__parameters__ << value if value.trainable?
                end
            end

            inv.each do |k|
                child = instance_variable_get(k)
                @__childlen__ << child if child.is_a?(Module)
            end

            @__flag_init_update__ = true
        end

        def parameters
            __update_childlen__
            return @__parameters__
        end
    end
end