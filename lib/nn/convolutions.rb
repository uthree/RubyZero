require_relative "./nn.rb"
require_relative "./convolution_utils.rb"

module RubyZero::NN
    Utils = RubyZero::ConvolutionUtils
    class Conv1d < Module
    private
        include Utils
    public
        def initialize(in_channels, out_channels, kernel_size, stride:1, padding:0)
            @in_channels = in_channels
            @out_channels = out_channels
            @kernel_size = kernel_size
            @stride = stride
            @padding = padding

            @filter = Linear.new(in_channels*kernel_size, out_channels)
            super()
        end

        def forward(x)
            cols = im2col_1d(x, @kernel_size, @padding, @stride)
            # [batch, width-kernel, channels*kernel]
            dist_shape = [cols.shape[0], cols.shape[1], @out_channels]
            flat_cols = cols.flatten(start_axis: 0, end_axis:1) # [batch*(width-kernel), channels*kernel]
            output_cols = @filter.call(flat_cols) # [batch*(width-kernel), out_channels]
            return output_cols.reshape(*dist_shape)
        end
    end

    class Conv2d < Module
    private
        include Utils
    public
        def initialize(in_channels, out_channels, kernel_size, stride:1, padding:0, kernel_height:nil, kernel_width:nil)
            @kernel_width = kernel_size if kernel_width.nil?
            @kernel_height = kernel_size if kernel_height.nil?
            @in_channels = in_channels
            @out_channels = out_channels
            @stride = stride
            @padding = padding

            @filter = Linear.new(in_channels*@kernel_width*@kernel_height, out_channels)
            super()
        end
        def forward(x)
            cols = im2col_2d(x, @kernel_height, @kernel_width, @padding, @stride)
            # [batch, width-kernel, height-kernel, channels*kernel]
            dist_shape = [cols.shape[0], cols.shape[1], cols.shape[2], @out_channels]
            flat_cols = cols.flatten(start_axis: 0, end_axis:2) # [batch*(width-kernel)*(height-kernel), channels*kernel]
            output_cols = @filter.call(flat_cols) # [batch*(width-kernel)*(height-kernel), out_channels]
            return output_cols.reshape(*dist_shape)
        end
    end

    class Conv3d < Module
    private
        include Utils
    public
        def initialize(in_channels, out_channels, kernel_size, stride:1, padding:0, kernel_depth:nil, kernel_height:nil, kernel_width:nil)
            @kernel_depth = kernel_size if kernel_depth.nil?
            @kernel_height = kernel_size if kernel_height.nil?
            @kernel_width = kernel_size if kernel_width.nil?
            @in_channels = in_channels
            @out_channels = out_channels
            @stride = stride
            @padding = padding

            @filter = Linear.new(in_channels*@kernel_depth*@kernel_height*@kernel_width, out_channels)
            super()
        end
        def forward(x)
            cols = im2col_3d(x, @kernel_depth, @kernel_height, @kernel_width, @padding, @stride)
            # [batch, width-kernel, height-kernel, depth-kernel, channels*kernel]
            dist_shape = [cols.shape[0], cols.shape[1], cols.shape[2], cols.shape[3], @out_channels]
            flat_cols = cols.flatten(start_axis: 0, end_axis:3) # [batch*(width-kernel)*(height-kernel)*(depth-kernel), channels*kernel]
            output_cols = @filter.call(flat_cols) # [batch*(width-kernel)*(height-kernel)*(depth-kernel), out_channels]
            return output_cols.reshape(*dist_shape)
        end
    end
end