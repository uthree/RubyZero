module RubyZero::Functions
    class ReShape < Function
        def initialize(*shape)
            @shape = shape
        end
        def forward(x)
            @old_shape = x.shape
            data = x.data.dup.reshape(*@shape)
            return Tensor.new(data)
        end
        def backward(dy)
            #p @old_shape
            #p dy.shape
            data = dy.data.dup.reshape(*@old_shape)
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

    # sum
    class Sum < Function
        def initialize(axis)
            @axis = axis
        end
        def forward(x)
            @repeats = x.shape[@axis]
            @pass_mode = x.ndim == 1
            data = x.data.sum(axis:@axis)
            return Tensor.new(data)
        end
        def backward(dy)
            if @pass_mode
                return [dy]
            else
                return [ dy.repeat(@repeats, axis: @axis) ]
            end
        end
    end

    # mean
    class Mean < Function
        def initialize(axis)
            @axis = axis
        end
        def forward(x)
            @repeats = x.shape[@axis]
            @pass_mode = x.ndim == 1
            data = x.data.sum(axis:@axis) / @repeats
            return Tensor.new(data)
        end
        def backward(dy)
            if @pass_mode
                return [dy]
            else
                dy = dy.repeat(@repeats, axis: @axis)
                dy_data = dy.data / @repeats
                return [Tensor.new(dy_data)]
            end
        end
    end

    # a.dot b
    class MatMul < Function
        def forward(a, b)
            data = a.data.dot(b.data)
            return Tensor.new(data)
        end
        def backward(dy)
            a,b = @inputs[0], @inputs[1]
            da = dy.data.dot(b.data.swapaxes(1,0))
            db = a.data.swapaxes(1,0).dot(dy.data)
            da = Tensor.new(da)
            db = Tensor.new(db)
            return [da, db]
        end
    end

    #slice
    class Slice < Function
        def initialize(range)
            @range = range
        end
        def forward(a)
            return Tensor.new(a.data[*@range])
        end
        def backward(dy)
            z = @inputs[0].data.new_zeros
            dy_data = dy.data
            z[*@range] = dy_data
            return [Tensor.new(z)]
        end
    end

    class Concatenate < Function
        def initialize(axis:0)
            @axis = axis
        end
        
        def forward(a, b)
            @a_shape = a.shape
            @b_shape = b.shape
            data = a.data.concatenate(b.data, axis: @axis)
            return Tensor.new(data)
        end

        def backward (dy)
            len_a = @a_shape[@axis]
            len_b = @b_shape[@axis]
            shape_a = Array.new(@a_shape.length, nil)
            shape_b = Array.new(@b_shape.length, nil)
            shape_a[@axis] = 0..len_a-1
            shape_b[@axis] = len_a..len_a+len_b-1
            da = dy.data[*shape_a]
            db = dy.data[*shape_b]
            return [ Tensor.new(da), Tensor.new(db) ]
        end
    end

end