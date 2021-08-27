module RubyZero::Core
    class Shape
        # Initialize a new shape
        # @param [Array<Axis>] axes
        # @return [Shape]
        def initialize(*axes)
            if axes.length > 0
                @axes = axes.map {|axis|
                    if axis.class == (Integer)
                        Axis.new(axis)
                    else
                        axis
                    end
                }
                @scaler = false
            else
                @axes = []
                @scaler = true
            end
        end

        def each
            @axes.each {|axis| yield axis}
        end
        include Enumerable

        attr_reader :axes,

        # update shape from narray
        # @param [Numo::Narray|Cumo::Narray] narray
        def update_shape_from_narray(narray)
            @axes = []
            if narray.shape == [1]
                @scaler = true
            else
                narray.shape.each_with_index do |size, axis|
                    @axes << Axis.new(size)
                end
                @scaler = false
            end
        end
        # @return [Integer]
        def ndim
            @axes.size
        end
        # @param [Integer|Object] key
        # @return [Axis]
        def get_axis(key)
            return @axes[key] if key.class == Integer
            return @axes.find {|axis| axis.name == key} if key.class != Integer
            return ArgumentError.new("invalid axis #{key}")
        end
        # @param [Integer|Object] key
        # @return [Axis]
        def [](key)
            get_axis(key)
        end
        
        def to_a()
            return [] if @scaler
            return @axes.map {|axis| axis.size} unless @scaler
        end
    end
end