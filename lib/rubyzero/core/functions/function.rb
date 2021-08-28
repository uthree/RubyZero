# Function (base class)
module RubyZero::Core::Functions
    class Function
        def initialize()
            @input = nil
        end
        def call(*args)
            @input = args
            @output = forward(*args)
            if @output.requires_grad
                @output.grad_function = self
            end
            return @output
        end
        def forward()
            raise NotImplementedError, "#{self.class}#forward not implemented."
        end
        def backward(dy)
            raise NotImplementedError, "#{self.class}#backward not implemented."
        end
    end
end