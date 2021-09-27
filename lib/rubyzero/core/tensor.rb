module RubyZero::Core
    class Tensor
        attr_accessor :data, :grad_fn, :grad
        def initialize(data, dtype: nil, device: RubyZero.device(:cpu))
            device = device
            @grad_fn = nil
            @grad = nil

            # check data type
            if data.is_a?(Array)
                if dtype
                    @data = dtype.get_type_on_calculator(device)[*data]
                else
                    @data = device.caluculator::NArray[*data]
                end
            elsif data.is_a?(Numeric)
                if dtype
                    @data = dtype.get_type_on_calculator(device)[data]
                else
                    @data = device.caluculator::NArray[data]
                end
            elsif data.is_a?(Numo::NArray)
                @data = data
            else 
                raise Execptions::TypeNotSupported, "Tensor data type not supported: #{data.class}"
            end
            @dtype = DataTypes.from_xmo_dtype(@data.class)
        end
        def dtype
            @dtype
        end
        def shape
            @data.shape
        end
        def inspect
            numo_inspect = @data.inspect.split("\n")[1..nil].join("\n")
            return "#{dtype}#shape=#{shape.to_a}\n#{numo_inspect}\ngrad_function=#{@grad_fn.class}"
        end
    end
end