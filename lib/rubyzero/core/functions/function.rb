require_relative '../exceptions.rb'

module RubyZero::Core::Functions
    class Function
        def initialize(*args, **kwargs, &block)
            
        end
        def forward(*args, **kwargs, &block)
            raise Execptions::NotImplementedError, "#{self.class}#forward() not implemented"
        end
        def backward(*args, **kwargs, &block)
            raise Execptions::NotImplementedError, "#{self.class}#backward() not implemented"
        end
        def call(*args)
            @inputs = args
            @output = forward(*args)
            if @output.requires_grad?
                @output.grad_fn = self
            end
            return @output
        end
    end
end