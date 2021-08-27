module RubyZero::Core
    class Shape
        attr_reader :axes
        # Initialize a new shape
        # @param [Array<Axis>] axes
        # @return [Shape]
        def initialize(*axes)
            @axes = axes
        end
        def ndim
            @axes.size
        end
        def each
            @axes.each {|axis| yield axis}
        end
        include Enumerable
    end
end