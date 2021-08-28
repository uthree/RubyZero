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
            return [SumZeroAxis.new().call(dy)]
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
            return [RepeatZeroAxis.new(dy.data.shape.to_a[0]).call(dy)]
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
            return [Transpose.new(*rev).call(dy)]
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
            return [ReShape.new(@input_shape).call(dy)]
        end
    end

    class SwapAxes < Function
        # @param [Integer] axis1
        # @param [Integer] axis2
        def initialize(axis1, axis2)
            @axis1 = axis1
            @axis2 = axis2
        end
        def forward(x)
            calculator = x.device.calculator
            data = x.data.swapaxes(@axis1, @axis2)
            return Tensor.new(data, device: x.device)
        end
        def backward(dy)
            return [SwapAxes.new(@axis2, @axis1).call(dy)]
        end
    end

    class MaxZeroAxis < Function
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            calculator = x.device.calculator
            @max_idx = x.data.argmax(axis: 0).to_i
            return Tensor.new(x.data[@max_idx], device: x.device)
        end
        def backward(dy)
            return [@input[@max_idx]]
        end
    end

    class MinZeroAxis < Function
        # @param [Tensor] x
        # @return [Tensor]
        def forward(x)
            calculator = x.device.calculator
            @min_idx = x.data.argmin(axis: 0).to_i
            return Tensor.new(data[@min_idx], device: x.device)
        end
        def backward(dy)
            return [@input[@min_idx]]
        end
    end
end

# Apply a function to tensor class.
module RubyZero::Core
    class Tensor
        # transpose tensor
        # @param [Array<Integer>|Shape] args
        def transpose(*args)
            if args[0].is_a?(Shape)
                args = args[0].to_a
            end
            return Functions::Transpose.new(*args).call(self)
        end

        # repeat tensor
        # @param [Integer] num_repeats
        # @option options [Integer] axis
        def repeat(num_repeats, axis: 0)
            repeated = Functions::RepeatZeroAxis.new(num_repeats).call(self)
            transposed = repeated.swap_axes(0, axis)
            return transposed
        end

        # caluculate sum of tensor
        # @param [Integer] axis
        # @return [Tensor]
        def sum(axis: 0)
            sza = Functions::SumZeroAxis.new().call(self)
            transposed = sza.swap_axes(0, axis)
            return transposed
        end

        # reshape tensor
        # @param [Array<Integer>|Shape] args
        # @return [Tensor]
        def reshape(*args)
            if args[0].is_a?(Shape)
                args = args[0].to_a
            end
            return Functions::Reshape.new(*args).call(self)
        end

        # swap axes of tensor
        # @param [Integer|Axis] axis1
        # @param [Integer|Axis] axis2
        # @return [Tensor]
        def swap_axes(axis1, axis2)
            axis1, axis2 = axis1.to_i, axis2.to_i
            if axis1 != axis2
                tensor = Functions::SwapAxes.new(axis1, axis2).call(self)
                new_shape = self.shape.swap_axes(axis1, axis2)
                tensor.apply_shape(new_shape)
                return tensor
            else
                return self
            end
        end

        # get min value of an axis
        # @param [Integer] axis
        # @return [Tensor]
        def max(axis: 0)
            return Functions::MaxZeroAxis.new().call(self.swap_axes(0, axis)).swap_axes(0, axis)
        end

        # get min value of an axis
        # @param [Integer] axis
        # @return [Tensor]
        def min(axis: 0)
            return Functions::MinZeroAxis.new().call(self.swap_axes(0, axis)).swap_axes(0, axis)
        end
    end
end