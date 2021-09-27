module RubyZero::Core
    class Tensor
        # initialize new tensor with given shape which filled with zeros.
        def self.zeros(shape, dtype: RubyZero::Core::DataTypes::Float64, device: RubyZero.device(:cpu))
            dtype = dtype.get_type_on_calculator(device)
            data = dtype.zeros(shape)
            Tensor.new(data, dtype:dtype, device:device)
        end

        def zeros_like()
            Tensor.zeros(self.shapped_data, dtype:self.dtype, device:self.device)
        end
    end

    module DataTypes
        class DataType
            def self.zeros(*args)
                Tensor.zeros(*args)
            end
        end
    end

    
end