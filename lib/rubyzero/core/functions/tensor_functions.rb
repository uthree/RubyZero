module RubyZero::Core::Functions
    class RepeatZeroAxis < Function
        # @param [Integer] num_repeats
        def initialize(num_repeats)
            @num_repeats = num_repeats
        end
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            calculator = x.device.calculator
            data = x.data.repeat(@num_repeats, axis: 0)
            return Tensor.new(data, device: x.device)
        end
        # @param [Tensor] dy
        def backward(dy)
            return SumZeroAxis.new().call(dy)
        end
    end

    class SumZeroAxis < Function
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            calculator = x.device.calculator
            data = x.data.sum(axis: 0)
            return Tensor.new(data, device: x.device)
        end
        # @param [Tensor] dy
        def backward(dy)
            return RepeatZeroAxis.new(dy.data.shape.to_a[0]).call(dy)
        end
    end

    class Transpose < Function
        # @param [Array<Integer>] args
        def initialize(*args)
            @args = args
        end
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            calculator = x.device.calculator
            data = x.data.transpose(*@args)
            t = Tensor.new(data, device: x.device)
            new_shape = t.shape.transpose(*@args)
            t.apply_shape(new_shape)
        end
        def backward(dy)
            rev = []
            @args.each_with_index do |i, idx|
                rev[i] = idx
            end
            return Transpose.new(*rev).call(dy)
        end
    end
end

# Apply a function to tensor class.
module RubyZero::Core
    class Tensor
        # transpose tensor
        # @param [Array<Integer>] args
        def transpose(*args)
            return RubyZero::Core::Functions::Transpose.new(*args).call(self)
        end
    end
end