module RubyZero::Core
    class Axis
        attr_reader :name, :length
        # Initialize Axis.
        # @param [Integer] length
        # @option kwargs [String] :name
        # @option kwargs [Array<String|Symbol|NilClass>] :keys
        # @return [Axis] 
        def initialize(length, **kwargs)
            name = kwargs[:name]
            keys = kwargs[:keys]

            @length = length
            @name = name
            @keys = keys
        end
        # Returns the index of the key.
        # @return [Integer]
        # @param [Object] key
        def key2index(key)
            if @keys[key]
                @keys.index(key)
            else
                raise ArgumentError, "key #{key} not found"
            end
        end
        # Returns the key of the index.
        # @return [Integer]
        # @param [Object] key
        def [](key)
            key2index(key)
        end
        # Returns length of the axis.
        # @return [Integer] 
        def size
            @length
        end
    end
end