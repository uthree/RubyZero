
module RubyZero
    module NN

    end

end

module RubyZero::NN
    class Module # template module of neural network
        attr_reader :__parameters__, :__childlen__, :mode
        def initialize()
            @__parameters__ = Parameters.new
            @__childlen__ = []
            @__flag_init_update__ = false
            @mode = :train
        end

        def call(*args, **kwargs, &block)
            args.each do |arg|
                arg.require_grad = true
            end
            forward(*args, **kwargs, &block)
        end

        def forward(*args, **kwargs, &block)
            raise NotImplementedError, "#{self.class}.forward method not implemented"
        end

        def eval()
            __update_childlen__()
            @mode = :eval
            @__parameters__.eval
            @__childlen__.each do |child|
                child.eval()
            end
            return nil
        end

        def train()
            __update_childlen__()
            @mode = :train
            @__parameters__.train
            @__childlen__.each do |child|
                child.train()
            end
            return nil
        end

        def __update_childlen__()
            @__childlen__ ||= []
            @__parameters__ = Parameters.new
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
                @__childlen__ << child if child.kind_of?(Module)
            end
            
            @__childlen__.each do |child|
                child.__update_childlen__
                @__parameters__ += child.parameters
            end

            @__flag_init_update__ = true
            return nil
        end

        def parameters
            __update_childlen__
            @__childlen__.uniq!
            @__parameters__.elements.uniq!
            return @__parameters__
        end
    end
end

