module RubyZero
    class Tensor
        def +(other)
            return Functions::Add.new().call(self, other)
        end

        def *(other)
            return Functions::Mul.new().call(self, other)
        end

        def -(other)
            return Functions::Sub.new().call(self, other)
        end

        def /(other)
            return Functions::Div.new().call(self, other)
        end

        def **(other)
            return Functions::Pow.new().call(self, other)
        end

        def -@
            return Functions::Neg.new().call(self)
        end
    end
end