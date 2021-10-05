module RubyZero::Core::Functions
    class Reshape < Function
        def initialize(shape)
            @dist_shape = shape
        end
        def forward(x1)
            new_arr = x1.data.reshape(@dist_shape)
            @prev_shape = x1.shape
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy.reshape(@prev_shape)]
        end
    end
    class SwapAxes < Function
        def initialize(axis1, axis2)
            @axis1 = axis1
            @axis2 = axis2
        end
        def forward(x1)
            new_arr = x1.data.swapaxes(@axis1, @axis2)
            @prev_shape = x1.shape
            new_t = RubyZero::Core::Tensor.new(new_arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy.swapaxes(@axis1, @axis2)]
        end
    end
    class Repeat < Function
        def initialize(axis, repeats)
            @axis = axis
            @repeats = repeats
        end
        def forward(x1)
            arr = x1.data
            arr = arr.reshape(*([1] + arr.shape))
            arr = arr.repeat(@repeats, axis:0)
            arr = arr.swapaxes(0, @axis)
            new_t = RubyZero::Core::Tensor.new(arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [ dy.sum(axis: @axis) ]
        end
    end

    class Sum < Function
        def initialize(axis)
            @axis = axis
        end
        def forward(x1)
            @repeats = x1.shape[@axis]
            arr = x1.data
            arr = arr.sum(axis: @axis)
            new_t = RubyZero::Core::Tensor.new(arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy.repeat(@repeats, axis: @axis) / @repeats]
        end
    end

    class Mean < Function
        def initialize(axis)
            @axis = axis
        end
        def forward(x1)
            @repeats = x1.shape[@axis]
            arr = x1.data
            arr = arr.mean(axis: @axis)
            new_t = RubyZero::Core::Tensor.new(arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            return [dy.repeat(@repeats, axis: @axis)]
        end
    end

    class DotProduct < Function
        def initialize()
        end
        def forward(x1, x2)
            arr = x1.data.dot(x2.data)
            new_t = RubyZero::Core::Tensor.new(arr, device: x1.device)
            return new_t
        end
        def backward(dy)
            x1, x2 = @inputs
            dx, dy = [dy.dot(x2.swapaxes(0,1)), x1.swapaxes(0,1).dot(dy)]
            return dx, dy
        end
    end
end

# apply Tensor class
module RubyZero::Core
    class Tensor
        def reshape(shape)
            return RubyZero::Core::Functions::Reshape.new(shape).call(self)
        end
        def swapaxes(axis1, axis2)
            return RubyZero::Core::Functions::SwapAxes.new(axis1, axis2).call(self)
        end
        def repeat(repeats, axis:0)
            return RubyZero::Core::Functions::Repeat.new(axis, repeats).call(self)
        end
        def sum(axis: 0)
            return RubyZero::Core::Functions::Sum.new(axis).call(self)
        end
        def mean(axis: 0)
            return RubyZero::Core::Functions::Mean.new(axis).call(self)
        end
        def dot(other)
            if other.is_a?(RubyZero::Core::Tensor)
                return RubyZero::Core::Functions::DotProduct.new().call(self, other)
            else
                return self*other
            end
        end
    end
end