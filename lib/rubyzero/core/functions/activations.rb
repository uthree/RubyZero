module RubyZero::Core::Functions
    class Sigmoid < Function
        # @option options [Float] :alpha gain
        def initialize(alpha: 1.0)
            @alpha = alpha
        end
        # @param [Tensor] x
        def forward(x)
            1.0 / (1.0 + F.exp(-@alpha * x))
        end
        def backward(dy)
            x = @input[0]
            return @alpha * F.exp(-@alpha * x) / (1.0 + F.exp(-@alpha * x))**2 * dy
        end
    end

    class ReLU
        
    end
end