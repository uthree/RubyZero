module RubyZero::Core
    module DataTypes
        class DType
            def to_s
                self.class.name
            end
            def ==(other)
                self.class == other.class
            end
            # @param [Device] device
            # @return [Class]
            def self.get_dtype_on_device(device)
                if device.id == :numo
                    if self == Boolean
                        return Numo::Bit
                    elsif self == Int8
                        return Numo::Int8
                    elsif self == Int16
                        return Numo::Int16
                    elsif self == Int32
                        return Numo::Int32
                    elsif self == Int64
                        return Numo::Int64
                    elsif self == UInt8
                        return Numo::UInt8
                    elsif self == UInt16
                        return Numo::UInt16
                    elsif self == UInt32
                        return Numo::UInt32
                    elsif self == UInt64
                        return Numo::UInt64
                    elsif self == Float32
                        return Numo::SFloat
                    elsif self == Float64
                        return Numo::DFloat
                    elsif self == RObject
                        return Numo::RObject
                    elsif self == Complex64
                        return Numo::SComplex
                    elsif self == Complex128
                        return Numo::DComplex
                    end
                end
            end
        end
        
        class Boolean < DType
        end
        class Int8 < DType
        end
        class Int16 < DType
        end
        class Int32 < DType
        end
        class Int64 < DType
        end
        class UInt8 < DType
        end
        class UInt16 < DType
        end
        class UInt32 < DType
        end
        class UInt64 < DType
        end
        class Float32 < DType
        end
        class Float64 < DType
        end
        class RObject < DType
        end
        class Complex64 < DType
        end
        class Complex128 < DType
        end

        def from_numo_dtype(dtype)
            case dtype
            when Numo::Bit
                Boolean
            when Numo::Int8
                Int8
            when Numo::Int16
                Int16
            when Numo::Int32
                Int32
            when Numo::Int64
                Int64
            when Numo::UInt8
                UInt8
            when Numo::UInt16
                UInt16
            when Numo::UInt32
                UInt32
            when Numo::UInt64
                UInt64
            when Numo::SFloat
                Float32
            when Numo::DFloat
                Float64
            when Numo::RObject
                RObject
            when Numo::SComplex
                Complex64
            when Numo::DComplex
                Complex128
            end
        end
    end
end