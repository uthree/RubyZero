module RubyZero::Core
    class Tensor
        # initialize new tensor with given shape which filled with zeros.
        def self.zeros(shape, dtype: RubyZero::Core::DataTypes::Float64, device: RubyZero.device(:cpu))
            dtype = dtype.get_type_on_calculator(device)
            data = dtype.zeros(shape)
            Tensor.new(data, dtype:dtype, device:device)
        end


        # initialize new tensor with given another tensor's shape which filled with zeros.
        def zeros_like()
            Tensor.zeros(self.shapped_data, dtype:self.dtype, device:self.device)
        end

        # initialize new tensor with given shape which filled with ones.
        def self.ones(shape, dtype: RubyZero::Core::DataTypes::Float64, device: RubyZero.device(:cpu))
            dtype = dtype.get_type_on_calculator(device)
            data = dtype.ones(shape)
            Tensor.new(data, dtype:dtype, device:device)
        end

        # initialize new tensor with given another tensor's shape which filled with ones.
        def ones_like()
            Tensor.ones(self.shape, dtype:self.dtype, device:self.device)
        end

        # initialize new tensor with given shape which filled with random values.
        def self.randn(shape, dtype: RubyZero::Core::DataTypes::Float64, device: RubyZero.device(:cpu))
            dtype = dtype.get_type_on_calculator(device)
            data = dtype.new(shape).rand_norm
            Tensor.new(data, dtype:dtype, device:device)
        end

        # initialize new tensor with given another tensor's shape which filled with random values.
        def randn_like()
            Tensor.randn(self.shape, dtype:self.dtype, device:self.device)
        end

        # initialize new tensor with given shape which filled with random values.
        def self.rand(shape, dtype: RubyZero::Core::DataTypes::Float64, device: RubyZero.device(:cpu))
            dtype = dtype.get_type_on_calculator(device)
            data = dtype.new(shape).rand
            Tensor.new(data, dtype:dtype, device:device)
        end

        # initialize new tensor with given another tensor's shape which filled with random values.
        def rand_like()
            Tensor.rand(self.shape, dtype: self.dtype, device:self.device)
        end

        def self.stack(tensors, axis:0)
            t = Tensor.new(tensors.map{|t| t.data})
            return t
        end
    end

    module DataTypes
        class DataType
            def self.zeros(*args)
                Tensor.zeros(*args, dtype: self)
            end
            def self.ones(*args)
                Tensor.ones(*args, dtype: self)
            end
            def self.randn(*args)
                Tensor.randn(*args, dtype: self)
            end
        end
    end
end

module RubyZero
    FloatTensor = RubyZero::Core::DataTypes::Float32
    DoubleTensor = RubyZero::Core::DataTypes::Float64
    IntTensor = RubyZero::Core::DataTypes::Int32
    LongTensor = RubyZero::Core::DataTypes::Int64
end