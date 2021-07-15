module RubyZero
    class Tensor
        def self.rand_norm(shape, mean:0, std:1, dtype:FloatTensor)
            data = dtype.get_type_numo.new(*shape).rand_norm(mean, std)
            return new data
        end
        def self.rand_uniform(shape, low:-1, high:1, dtype:FloatTensor)
            data = dtype.get_type_numo.new(*shape).rand_uniform(low, high)
            return new data
        end
    end
end