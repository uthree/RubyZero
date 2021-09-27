module RubyZero::Core
    class Tensor
        def +(other)
            RubyZero::Core::Functions::Add.new().call(self, other)
        end
        def -(other)
            RubyZero::Core::Functions::Sub.new().call(self, other)
        end
        def *(other)
            if other.is_a?(Numeric)
                RubyZero::Core::Functions::MulScalar.new(other).call(self)
            else
                RubyZero::Core::Functions::Mul.new().call(self, other)
            end
        end
        def /(other)
            if other.is_a?(Numeric)
                RubyZero::Core::Functions::DivScalar.new(other).call(self)
            else
                RubyZero::Core::Functions::Div.new().call(self, other)
            end
        end
    end
end