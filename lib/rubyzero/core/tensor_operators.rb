module RubyZero::Core
    class Tensor
        def +(other)
            RubyZero::Core::Functions::Add.new().call(self, other)
        end
    end
end