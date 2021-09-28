require_relative '../exceptions.rb'

module RubyZero::Core::Functions
    # Function class
    class Function
        attr_reader :inputs, :output
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
            if @inputs.any?{|t| t.requires_grad?}
                @output.grad_fn = self
                @output.requires_grad = true
            end
            return @output
        end
        def inspect
            return "#<#{self.class}>"
        end
    end
end