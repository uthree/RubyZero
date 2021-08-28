module RubyZero::Core
    class Axis
        attr_reader :name, :length
        # Initialize Axis.
        # @param [Integer] length
        # @option kwargs [String] :name
        # @option kwargs [Array<String|Symbol|NilClass>] :keys
        # @return [Axis] 
        def initialize(length, parent, **kwargs)
            name = kwargs[:name]
            keys = kwargs[:keys]
            
            @length = length
            @name = name
            @keys = keys
            @parent = parent
        end
        # Returns the index of the key.
        # @return [Integer|Range]
        # @param [Object|Range<Object, Object>] key
        def key2index(key)
            if keys.kind_of?(Range)
                b = key.begin
                e = key.end
                s = key.step
                b_i = @keys[b]
                e_i = @keys[e]
                rng = Range.new(b_i, e_i).step(s)

            elsif @keys[key]
                @keys.index(key)
            else
                raise ArgumentError, "key #{key} not found"
            end
        end
        # Returns the key of the index.
        # @return [Integer|Range]
        # @param [Object] key
        def [](key)
            key2index(key)
        end
        # Returns length of the axis.
        # @return [Integer] 
        def size
            @length
        end

        def index
            return @parent.index
        end

        def to_i
            return index
        end
    end
end