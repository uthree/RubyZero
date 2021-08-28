module RubyZero::Core
    class Tensor
        attr_reader :shape, :dtype
        attr_accessor :grad_function, :grad_tensor, :requires_grad, :data, :device

        # @param [Shape] shape
        # @param [Datatypes::DType] dtype
        # @param [Device] device
        def initialize(data=[], shape:Shape.new(), dtype: nil, device:Device.new(:numo))
            if data
                if data.is_a?(Array)
                    data = Numo::NArray[*data]
                    predicted_type = DataTypes::Float64
                    if data.flatten[0].is_a?(Integer)
                        predicted_type = DataTypes::Int64
                    elsif data.flatten[0].is_a?(Float)
                        predicted_type = DataTypes::Float64
                    elsif data.flatten[0].is_a?(Complex)
                        predicted_type = DataTypes::Complex128
                    else
                        predicted_type = DataTypes::RObject
                    end
                    @dtype ||= predicted_type
                elsif data.is_a?(Numo::NArray)
                    @dtype ||= DataTypes::from_numo_dtype(data.class)
                elsif data.is_a>(Tensor)
                    return new(data.data, shape: data.shape, dtype: data.dtype)
                else
                    raise TypeError, "data must be Array or Numo::NArray"
                end
                @device = device
                @data = data
                @shape = Shape.new(*data.shape)
                @dtype ||= DataTypes::from_numo_dtype(data.class)
            else
                @device = device
                @shape = shape
                @dtype ||= DataTypes::RObject
                @data = @dtype.get_dtype_on_device(@device).zeros(*@shape.to_a)
            end
            @grad_function = nil
            @grad_tensor = nil
            @requires_grad = true
            return Functions::Constant.new.call(self)
        end
        # Execute gradient function.
        # @return [RubyZero::Core::Tensor]
        def backward()
            @grad_tensor = ones_like()
            if self.requires_grad and @grad_function
                grad_result = @grad_function.backward(@grad_tensor)
                self.grad_function.input.each_with_index {|t, i| t.grad_tensor = grad_result[i] }
            end
            return self
        end

        # @retrun [String]
        def inspect
            numo_inspect = @data.inspect.split("\n")[1..nil].join("\n")
            return "#{dtype}#shape=#{shape.to_a}\n#{numo_inspect}\ngrad_function=#{@grad_function.class}"
        end

        # @return [Integer]
        def ndim
            return self.shape.ndim
        end

        # @param [Datatypes::DType] dtype
        def cast_to(dtype)
            @data = dtype.get_dtype_on_device(@device).cast(@data)
            if @grad_tensor and @grad_tensor.dtype != dtype and @grad_tensor != self
                @grad_tensor.cast_to(dtype)
            end
            nil
        end

        # initialize RubyZero::Core::Tensor[1, 2, 3... ] style.
        # @param [Array<Object>] data
        # @return [RubyZero::Core::Tensor]
        def self.[](*data)
            new(data)
        end

        # @return [Array<Object>]
        def to_a
            return self.data.to_a
        end

        # Initialize zeros tensor.
        # @param [Shape|Array<Integer>] shape
        # @param [Datatypes::DType] dtype
        # @option options [Device] :device
        # @return [RubyZero::Core::Tensor]
        def self.zeros(shape, dtype, device:Device.new(:numo))
            data = dtype.get_dtype_on_device(device).zeros(*shape.to_a)
            t = new(data, shape: shape, dtype: dtype)
            return t
        end
        # Initialize ones tensor.
        # @param [Shape|Array<Integer>] shape
        # @param [Datatypes::DType] dtype
        # @option options [Device] :device
        # @return [RubyZero::Core::Tensor]
        def self.ones(shape, dtype, device:Device.new(:numo))
            data = dtype.get_dtype_on_device(device).ones(*shape.to_a)
            t = new(data, shape: shape, dtype: dtype)
            return t
        end

        # Initialize tensor with other tensor's shape, dtype, and device. witch data is zeros.
        # @param [RubyZero::Core::Tensor] tensor
        # @return [RubyZero::Core::Tensor]
        def self.zeros_like(tensor)
            shape, dtype, device = tensor.shape, tensor.dtype, tensor.device
            data = dtype.get_dtype_on_device(device).zeros(*shape.to_a)
            t = new(data, shape: shape, dtype: dtype, device: device)
            return t
        end
        # Initialize tensor with other tensor's shape, dtype, and device. witch data is ones.
        # @param [RubyZero::Core::Tensor] tensor
        # @return [RubyZero::Core::Tensor]
        def self.ones_like(tensor)
            shape, dtype, device = tensor.shape, tensor.dtype, tensor.device
            data = dtype.get_dtype_on_device(device).ones(*shape.to_a)
            t = new(data, shape: shape, dtype: dtype, device: device)
            return t
        end

        
        # Initialize tensor with same shape, same dtype, and same device. witch data is zeros.
        # @return [RubyZero::Core::Tensor]
        def zeros_like
            return self.class.zeros_like(self)
        end

        # Initialize tensor with same shape, same dtype, and same device. witch data is ones.
        # @return [RubyZero::Core::Tensor]
        def ones_like
            return self.class.ones_like(self)
        end
    end
end

module TensorInitializer
    Tensor = RubyZero::Core::Tensor
    FloatTensor = RubyZero::Core::DataTypes::Float32
    IntTensor = RubyZero::Core::DataTypes::Int32
    LongTensor = RubyZero::Core::DataTypes::Int64
    DoubleTensor = RubyZero::Core::DataTypes::Float64
end

module RubyZero
    include TensorInitializer
end