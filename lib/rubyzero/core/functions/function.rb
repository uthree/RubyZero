require_relative '../exceptions.rb'

module RubyZero::Functions
    class Function
        def initialize(*args, **kwargs, &block)
            
        end
        def forward(*args, **kwargs, &block)
            raise Execptions::NotImplementedError, "Function::forward() not implemented"
        end
        def backward(*args, **kwargs, &block)
            raise Execptions::NotImplementedError, "Function::backward() not implemented"
        end
        def call(*args)
            @inputs = args
            @output = forward(*args)
            return @output
        end
    end
end