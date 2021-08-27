module RubyZero::Core
    class Device
        attr_reader :id, :calculator
        def initialize(id = :numo)
            if id == :numo
                @id = :numo
                @calculator = Numo
            end
        end
    end
end