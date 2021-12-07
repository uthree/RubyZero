module RubyZero::Core
    class Device
        attr_reader :caluculator, :identifier
        def initialize(identifier)
            sym = identifier.to_sym
            if sym == :cpu
                @caluculator = Numo
                @identifier = sym
            else
                raise RubyZero::Core::Exceptions::DeviceNotSupported, "Device #{identifier} is not supported."
            end
        end
        def xmo
            if @identifier == :cpu
                return Numo
            end
        end
    end
end

module RubyZero
    @@ident_device = {}
    def self.device(identifier)
        if @@ident_device[identifier]
            return @@ident_device[identifier]
        else
            d = Core::Device.new(identifier)
            @@ident_device[identifier] = d
            return d
        end
    end
end