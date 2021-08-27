module RubyZero::Core
    class Tensor
        attr_reader :shape, :dtype, :data

        # @param [Shape] shape
        # @param [Datatypes::DType] dtype
        # @param [Device] device
        def initialize(shape, dtype, device)
            @device = device
            @shape = shape
            @dtype = dtype
            p "DTYPE IS"
            p @dtype.get_dtype_on_device(@device)
            @data = @dtype.get_dtype_on_device(@device).zeros(*@shape.to_a)
        end
        # @param [Datatypes::DType] dtype
        def cast_to(dtype)
            @data = dtype.get_dtype_on_device(@device).cast(@data)
        end

        def zeros
            @dtype.get_dtype_on_device(@device).zeros(*@shape.to_a)
        end
    end
end