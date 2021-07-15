module RubyZero
    module Functions

    end
end

module RubyZero::Functions
    class Function # virtual class
        attr_accessor :inputs, :output
        def initialize(*hyper_params)
        end

        def call(*inputs)
            @inputs = inputs
            output = forward(*inputs)
            if @inputs.all?{|t| t.require_grad}
                output.grad_fn = self
            end
            @output = output
            return output
        end

        def forward(*inputs)
            raise NotImplementedError "#{self.class}.forward is not implemented."
        end

        def backward(*inputs)
            raise NotImplementedError "#{self.class}.backward is not implemented."
        end
    end

    # operators
    # a+b
    class Add < Function
        def forward(a, b)
            return Tensor.new(a.data + b.data)
        end
        def backward(dy)
            return dy, dy
        end
    end

    # -a
    class Neg < Function
        def forward(a)
            return Tensor.new(-a.data)
        end
        def backward(dy)
            return [-dy]
        end
    end

    # a-b
    class Sub < Function
        def forward(a, b)
            return Tensor.new(a.data - b.data)
        end
        def backward(dy)
            return [dy, -dy]
        end
    end

    # a*b
    class Mul < Function
        def forward(a, b)
            return Tensor.new(a.data * b.data)
        end
        def backward(dy)
            return [dy * @inputs[1], dy * @inputs[0]]
        end
    end

    # a/b
    class Div < Function
        def forward(a, b)
            return Tensor.new(a.data / b.data)
        end
        def backward(dy)
            da = dy.data / @inputs[1].data 
            db = -dy.data / @inputs[0].data ** 2
            da = Tensor.new(da)
            db = Tensor.new(db)
            return [da, db]
        end
    end

    # a**b
    class Pow < Function
        def forward(a, b)
            return Tensor.new(a.data ** b.data)
        end
        def backward(dy)
            a = @inputs[0].data
            b = @inputs[1].data
            da = dy.data * b * a ** (b - 1)
            db = dy.data * a ** b * dy.xm::NMath.log(a)
            da = Tensor.new(da)
            db = Tensor.new(db)
            return [da, db]
        end
    end
end