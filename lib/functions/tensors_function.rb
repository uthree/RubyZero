module RubyZero::Functions
    class ReShape < Function
        def initialize(*shape)
            @shape = shape
        end
        def forward(x)
            @old_shape = x.shape
            x.data.reshape(*@shape)
            return Tensor.new(x)
        end
        def backward(dy)
            data = dy.data.reshape(*@old_shape)
            return [Tensor.new(data)]
        end
    end

    class Transpose < Function
        def initialize(*axes)
            @axis = axes
        end
        def forward(x)
            data = x.data.transpose(*@axes)
            return Tensor.new(data)
        end
        def backward(dy)
            data = dy.data.transpose(*@axes)
            return [Tensor.new(data)]
        end
    end

    class SwapAxes < Function
        def initialize(axis1, axis2)
            @axis1 = axis1
            @axis2 = axis2
        end
        def forward(x)
            data = x.data.swapaxes(@axis1, @axis2)
            return Tensor.new(data)
        end
        def backward(dy)
            data = dy.data.swapaxes(@axis1, @axis2)
            return [Tensor.new(data)]
        end
    end

    # Numoにはおそらく実装されていなかったので作成。
    # axis=0方向に要素をrepeats個繰り返したものを返す。
    class RepeatZeroAxis < Function
        def initialize(repeats)
            @repeats = repeats
        end

        def forward(x)
            old_shape = x.shape
            data = x.data
            new_shape = old_shape.dup.insert(0, @repeats).reverse
            new_data = data.repeat(@repeats)
            new_data = new_data.reshape(*new_shape)
            new_axes = (new_data.ndim-1).downto(0).to_a
            new_data = new_data.transpose(*new_axes)
            return Tensor.new(new_data)
        end

        def backward(dy)
            return [ Tensor.new(dy.data.sum(0)) ]
        end
    end

    class Sum < Function
        def initialize(axis)
            @axis = axis
        end
        def forward(x)
            @repeats = x.shape[@axis]
            data = x.data.sum(axis:@axis)
            return Tensor.new(data)
        end
        def backward(dy)
            return [ dy.repeat(@repeats, axis: @axis) ]
        end
    end
end