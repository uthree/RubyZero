module RubyZero::Core
    module DataTypes
        # convert Rubyzero datatypes from Numo/Cumo classes.
        def self.from_xmo_dtype(klass)
            case klass.name 
            when "Numo::NArray"
                return Float64
            when "Numo::Bit"
                return Boolean
            when "Numo::Int8"
                return Int8
            when "Numo::Int16"
                return Int16
            when "Numo::Int32"
                return Int32
            when "Numo::Int64"
                return Int64
            when "Numo::UInt8"
                return UInt8
            when "Numo::UInt16"
                return UInt16
            when "Numo::UInt32"
                return UInt32
            when "Numo::UInt64"
                return UInt64
            when "Numo::SFloat"
                return Float32
            when "Numo::DFloat"
                return Float64
            when "Numo::SComplex"
                return Complex64
            when "Numo::DComplex"
                return Complex128
            end
        end

        class DataType
            def self.get_type_on_calculator(device)
                device.caluculator
            end
            def self.[](*data)
                return Tensor.new(data, dtype:self, device: RubyZero.device(:cpu))
            end
        end
        class Boolean < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::Bit
            end
        end
        class Int8 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::Int8
            end
        end
        class Int16 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::Int16
            end
        end
        class Int32 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::Int32
            end
        end
        class Int64 < DataType
            def self.get_type_on_calculator
                device.caluculator::Int64
            end
        end
        class UInt8 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::UInt8
            end
        end
        class UInt16 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::UInt16
            end
        end
        class UInt32 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::UInt32
            end
        end
        class UInt64 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::UInt64
            end
        end
        class Float32 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::SFloat
            end
        end
        class Float64 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::DFloat
            end
        end
        class Complex64 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::SComplex
            end
        end
        class Complex128 < DataType
            def self.get_type_on_calculator(device)
                device.caluculator::DComplex
            end
        end
    end
end