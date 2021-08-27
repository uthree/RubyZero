module RubyZero::Core
    class Tensor
        attr_reader :shape, :dtype

        # @param [Shape] shape
        # @param [Datatypes::DType] dtype
        # @param [Device] device
        def initialize(data=[], shape:Shape.new(), dtype: nil, device:Device.new(:cpu))
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
                    dtype ||= predicted_type
                elsif data.is_a?(Numo::NArray)
                    dtype ||= data.class
                else
                    raise TypeError, "data must be Array or Numo::NArray"
                end

                @device = device
                @data = data
                @shape = Shape.new(*data.shape)
                @dtype = dtype
            else
                @device = device
                @shape = shape
                @dtype = dtype
                p "DTYPE IS"
                p @dtype.get_dtype_on_device(@device)
                @data = @dtype.get_dtype_on_device(@device).zeros(*@shape.to_a)
            end
        end

        # @param [Datatypes::DType] dtype
        def cast_to(dtype)
            @data = dtype.get_dtype_on_device(@device).cast(@data)
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
    end
end

module TensorInitializer
    Tensor = RubyZero::Core::Tensor
end

module RubyZero
    include TensorInitializer
end