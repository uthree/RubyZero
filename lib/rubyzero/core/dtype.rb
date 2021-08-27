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
            def get_dtype_on_device(device)
                if device.id == :numo
                    case self.class 
                    when Boolean
                        return Numo::Bit
                    when Int8
                        return Numo::Int8
                    when Int16
                        return Numo::Int16
                    when Int32
                        return Numo::Int32
                    when Int64
                        return Numo::Int64
                    when UInt8
                        return Numo::UInt8
                    when UInt16
                        return Numo::UInt16
                    when UInt32
                        return Numo::UInt32
                    when UInt64
                        return Numo::UInt64
                    when Float32
                        return Numo::SFloat
                    when Float64
                        return Numo::DFloat
                    when RObject 
                        return Numo::RObject
                    when Complex64
                        return Numo::SComplex
                    when Complex128
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
    end
end