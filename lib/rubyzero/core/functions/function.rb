require_relative '../exceptions.rb'
require 'unicode_plot'

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
            @inputs = args unless @__without_save_inputs__
            output = forward(*args)
            if @inputs.any?{|t| t.requires_grad?}
                output.grad_fn = self
                output.requires_grad = true
            end
            @output = output unless @__without_save_output__
            return output
        end
        def inspect
            return "#<#{self.class}>"
        end
        def self.plot(range: (-10..10).step(0.01))
            inputs = range.to_a
            outputs = inputs.map{|t| self.new().call(RubyZero::Core::Tensor.new(t)).data[0]}
            plot = UnicodePlot.lineplot(inputs, outputs, name: self.name)
            plot.render
        end
    end
end