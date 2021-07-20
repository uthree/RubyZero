require_relative "./nn.rb"

module RubyZero
    module ConvolutionUtils
        def im2col_1d(x, kernel, padding)
            # x: [batch, width, channels]
            # padding
            x = padding_1d(x, padding)
            cols = []
            width = x.shape[1]
            kernel.times do |i|
                col = x[nil, i..i+width-kernel, nil]
                cols << col
            end
            x = Tensor.stack(cols, axis: 2)
            # [batch, width-kernel, channels*kernel]
            return x
        end

        def padding_1d(x, padding) # x: [batch, width, channels]
            if padding > 0
                length = x.shape[1]-1
                padded_len = length + 2 * padding
                new_shape = [x.shape[0], padded_len, x.shape[2]]
                t = Tensor.zeros(new_shape, x.dtype)
                t[nil, padding..length, nil] = x
                return t
            else 
                return x
            end
        end

        def im2col_2d(x, kernel_h, kernel_w, padding)
            # x: [batch, height, width, channels]
            # padding
            x = padding_2d(x, padding)
            cols = []
            height = x.shape[1]
            width = x.shape[2]
            kernel_h.times do |i|
                kernel_w.times do |j|
                    col = x[nil, i..i+height-kernel_h, j..j+width-kernel_w, nil]
                    cols << col
                end
            end
            x = Tensor.stack(cols, axis: 3)
            # [batch, height-kernel_h, width-kernel_w, channels*kernel_h*kernel_w]
            return x
        end

        def padding_2d(x, padding)
            # x: [batch, height, width, channels]
            if padding > 0
                h,     w     = x.shape[1], x.shape[2]
                pad_h, pad_w = x.shape[1] + 2 * padding, x.shape[2] + 2 * padding
                new_shape = [x.shape[0], pad_h, pad_w, x.shape[3]]
                t = Tensor.zeros(new_shape, x.dtype)
                t[nil, pad_h..h, pad_w..w, nil] = x
                return t
            else
                return x
            end
        end

        def padding_3d(x, padding)
            # x: [batch, depth, height, width, channels]
            if padding > 0
                d,     h,     w     = x.shape[1], x.shape[2], x.shape[3]
                pad_d, pad_h, pad_w = x.shape[1] + 2 * padding, x.shape[2] + 2 * padding, x.shape[3] + 2 * padding
                new_shape = [x.shape[0], pad_d, pad_h, pad_w, x.shape[4]]
                t = Tensor.zeros(new_shape, x.dtype)
                t[nil, pad_d..d, pad_h..h, pad_w..w, nil] = x
                return t
            else
                return x
            end
        end

        def im2col_3d(x, kernel_d, kernel_h, kernel_w, padding)
            # x: [batch, depth, height, width, channels]
            # padding
            x = padding_3d(x, padding)
            cols = []
            depth = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
            kernel_d.times do |i|
                kernel_h.times do |j|
                    kernel_w.times do |k|
                        col = x[nil, i..i+depth-kernel_d, j..j+height-kernel_h, k..k+width-kernel_w, nil]
                        cols << col
                    end
                end
            end
            x = Tensor.stack(cols, axis: 4)
            # [batch, depth-kernel_d, height-kernel_h, width-kernel_w, channels*kernel_d*kernel_h*kernel_w]
            return x
        end
    end
end