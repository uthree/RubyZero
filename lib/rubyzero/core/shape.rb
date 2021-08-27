module RubyZero::Core
    class Shape
        def each
            @axes.each {|axis| yield axis}
        end
        include Enumerable

        attr_reader :axes
        # Initialize a new shape
        # @param [Array<Axis>] axes
        # @return [Shape]
        def initialize(*axes)
            @axes = axes
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
            return TypeError.new("invalid axis #{key}")
        end
        def to_a()
            return @axes.map {|axis| axis.size}
        end
    end
end