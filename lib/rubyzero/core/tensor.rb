module RubyZero::Core
    # Tensor class
    class Tensor
        attr_accessor :data, :grad_fn, :grad, :device, :requires_grad
        def initialize(data, dtype: nil, device: nil)
            @device = device || RubyZero.device(:cpu)
            @grad_fn = nil
            @grad = nil
            @requires_grad = false

            # check data type
            if data.is_a?(Array)
                if dtype
                    @data = dtype.get_type_on_calculator(device)[*data]
                else
                    @data = @device.caluculator::NArray[*data]
                end
            elsif data.is_a?(Numeric)
                if dtype
                    @data = dtype.get_type_on_calculator(device)[data]
                else
                    @data = @device.caluculator::NArray[data]
                end
            elsif data.is_a?(Numo::NArray)
                @data = data
            else 
                raise Execptions::TypeNotSupported, "Tensor data type not supported: #{data.class}"
            end
            @dtype = DataTypes.from_xmo_dtype(@data.class)
        end
        # get data type
        # @return [DataTypes::DataType]
        def dtype
            @dtype
        end
        # get shape
        # @return [Array<Integer>]
        def shape
            @data.shape
        end
        # get tensor summary
        # @return [String]
        def inspect
            numo_inspect = @data.inspect.split("\n")[1..nil].join("\n")
            return "#{dtype}#shape=#{shape.to_a}\n#{numo_inspect}\ngrad_fn=#{@grad_fn.class}"
        end
        # @return [Boolean]
        def requires_grad?
            @requires_grad
        end
        # detach from graph
        def detach
            @grad_fn = nil
        end
    end
end