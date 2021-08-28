module RubyZero::Core::Functions
    class Constant < Function
        # Initialize a new constant tensor
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            return x
        end
        def backward(dy)
            return [Tensor.ones_like(@input[0]) * dy]
        end
    end

    class Negative < Function
        # Negate a number
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            Tensor.new(-x.data)
        end
        def backward(dy)
            return [-dy]
        end
    end

    class Add < Function
        # Add two tensors
        # @param [Tensor] x
        # @param [Tensor] y
        # @return [Tensor]
        def forward(x, y)
            Tensor.new(x.data + y.data)
        end
        def backward(dy)
            return [dy, dy]
        end
    end

    class Sub < Function
        # Subtract two tensors
        # @param [Tensor] x
        # @param [Tensor] y
        # @return [Tensor]
        def forward(x, y)
            Tensor.new(x.data - y.data)
        end
        def backward(dy)
            return [dy, -dy]
        end
    end

    class Mul < Function
        # Multiply two tensors
        # @param [Tensor] x
        # @param [Tensor] y
        # @return [Tensor]
        def forward(x, y)
            Tensor.new(x.data * y.data)
        end
        def backward(dy)
            return [dy * @input[1], dy * @input[0]]
        end
    end

    class Div < Function
        # Divide two tensors
        # @param [Tensor] x
        # @param [Tensor] y
        # @return [Tensor]
        def forward(x, y)
            Tensor.new(x.data / y.data)
        end
        def backward(dy)
            return [dy / @input[1], -dy * @input[0] / (@input[1] ** 2)]
        end
    end
end

# Apply a operator functions to a Tensor class
module RubyZero::Core
    class Tensor
        # @param [Tensor] other
        # @return [Tensor]
        def +(other)
            RubyZero::Core::Functions::Add.new.call(self, other)
        end
        # @param [Tensor] other
        # @return [Tensor]
        def -(other)
            RubyZero::Core::Functions::Sub.new.call(self, other)
        end
        # @param [Tensor] other
        # @return [Tensor]
        def *(other)
            RubyZero::Core::Functions::Mul.new.call(self, other)
        end
        # @param [Tensor] other
        # @return [Tensor]
        def /(other)
            RubyZero::Core::Functions::Div.new.call(self, other)
        end
        # @return [Tensor]
        def -@
            RubyZero::Core::Functions::Negative.new.call(self)
        end
    end
end
