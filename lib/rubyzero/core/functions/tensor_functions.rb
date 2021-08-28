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
            data = x.data
            data = data.reshape(*([1]+(data.shape.to_a)))
            data = data.repeat(@num_repeats, axis: 0)
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
            return t
        end
        def backward(dy)
            rev = []
            @args.each_with_index do |i, idx|
                rev[i] = idx
            end
            return Transpose.new(*rev).call(dy)
        end
    end

    class Reshape < Function
        # @param [Array<Integer>|Shape] args
        def initialize(*args)
            if args.length == 1 and args[0].is_a?(Shape)
                @dist_shape = args[0].to_a
            else
                @dist_shape = args
            end
        end
        def forward(x)
            @input_shape = x.shape
            data = x.data.reshape(*@dist_shape)
            return Tensor.new(data, device: x.device)
        end
        def backward(dy)
            return ReShape.new(@input_shape).call(dy)
        end
    end
end

# Apply a function to tensor class.
module RubyZero::Core
    class Tensor
        # transpose tensor
        # @param [Array<Integer>] args
        def transpose(*args)
            return Functions::Transpose.new(*args).call(self)
        end

        # repeat tensor
        # @param [Integer] num_repeats
        # @option options [Integer] axis
        def repeat(num_repeats, axis: 0)
            repeated = Functions::RepeatZeroAxis.new(num_repeats).call(self)
            transposed = Functions::Transpose.new(0, axis+1).call(repeated)
            return transposed
        end

        # caluculate sum of tensor
        # @param [Integer] axis
        # @return [Tensor]
        def sum(axis: 0)
            sza = Functions::SumZeroAxis.new().call(self)
            transposed = Functions::Transpose.new(0, axis).call(sza)
            return transposed
        end

        # reshape tensor
        # @param [Array<Integer>|Shape] args
        # @return [Tensor]
        def reshape(*args)
            return Functions::Reshape.new(*args).call(self)
        end
    end
end