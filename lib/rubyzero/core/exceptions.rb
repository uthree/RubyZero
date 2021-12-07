module RubyZero::Core
    module Exceptions
        class NoInplementError < StandardError ; end
        class DeviceNotSupported < StandardError ; end
        class TypeNotSupported < StandardError ; end
    end
end