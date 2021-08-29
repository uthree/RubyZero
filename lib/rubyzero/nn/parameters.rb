module RubyZero::NN
    class Parameter
        attr_reader :parent, :value
        # Initialize parameter
        # @param [Tensor] tensor
        def initialize(tensor)
            @value = tensor
            @value.requires_grad = true
        end
    end

    # Stack of Parameters.
    class Parameters
        def initialize()
            @parameters = []
        end
        # Add parameter to stack
        # @param [Tensor|Parameter] tensor
        def <<(parameter)
            if parameter.class == Parameter
                @parameters << parameter
            elsif parameter.class == Tensor
                @parameters << Parameter.new(parameter)
            else
                raise TypeError, "Expected Parameter or Tensor, got #{parameter.class}"
            end
        end
        def +(other)
            if other.class == Parameters
                params = Parameters.new()
                (@parameters + other.elements).each do |parameter|
                    params << parameter
                end
                return params
            else
                raise TypeError, "Expected Parameters, got #{other.class}"
            end
        end
        def index(parameter)
            @parameters.index(parameter)
        end
        def each(&block)
            @parameters.each do |parameter|
                block.call(parameter)
            end
        end
        def elements
            return @parameters
        end
        def __number_of_parameters__
            sum = 0
            @parameters.each do |parameter|
                sum += parameter.value.shape.to_a.inject(:*)
            end
            return sum
        end
        include Enumerable
    end
end