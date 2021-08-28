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
            return [@alpha * F.exp(-@alpha * x) / (1.0 + F.exp(-@alpha * x))**2 * dy]
        end
    end

    class ReLU < Function
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            @pass_through = x > 0
            return x * @pass_through
        end
        def backward(dy)
            return [dy * @pass_through]
        end
    end

    class LeakyReLU < Function
        # @param [Tensor] x
        # @param [Float] alpha
        def initialize(alpha: 0.01)
            @alpha = alpha
        end
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            @pass_through = x > 0
            return x * @pass_through + @alpha * (1.0 - @pass_through) * x
        end
        def backward(dy)
            return [dy * (@pass_through + @alpha * (1.0 - @pass_through))]
        end
    end

    class SoftmaxZeroAxis < Function
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            x_max = x.max(axis: 0)
            x_exp = F.exp(x - x_max)
            @x_exp_sum = x_exp.sum(axis: 0)
            return @x_exp / @x_exp_sum
        end
        def backward(dy)
            return [F.exp(dy) / @x_exp_sum]
        end
    end
end