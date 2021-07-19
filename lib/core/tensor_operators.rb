module RubyZero
    class Tensor
        def +(other)
            other = Tensor.new other unless other.is_a?(Tensor)
            other = other.repeat_to(self)
            return Functions::Add.new().call(self, other)
        end

        def *(other)
            other = Tensor.new other unless other.is_a?(Tensor)
            other = other.repeat_to(self)
            return Functions::Mul.new().call(self, other)
        end

        def -(other)
            other = Tensor.new other unless other.is_a?(Tensor)
            other = other.repeat_to(self)
            return Functions::Sub.new().call(self, other)
        end

        def /(other)
            other = Tensor.new other unless other.is_a?(Tensor)
            other = other.repeat_to(self)
            return Functions::Div.new().call(self, other)
        end

        def **(other)
            other = Tensor.new other unless other.is_a?(Tensor)
            other = other.repeat_to(self)
            return Functions::Pow.new().call(self, other)
        end

        def -@
            return Functions::Neg.new().call(self)
        end

        # slice
        def [](*args)
            return Functions::Slice.new(args).call(self)
        end
    end
end