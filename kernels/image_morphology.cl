__kernel void image_process(__read_only image2d_t inputImg, __write_only image2d_t outputImg)
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE |
                                   CLK_ADDRESS_REPEAT          |
                                   CLK_FILTER_NEAREST;

    int2 currentPosition = (int2)(get_global_id(0), get_global_id(1));

    uint4 currentPixel = (uint4)(0,0,0,0);
    currentPixel = read_imageui(inputImg, imageSampler, currentPosition);

    currentPixel.x = 255;
    currentPixel.y = 255;
    currentPixel.z = 255;
    currentPixel.w = 255;

    write_imageui(outputImg, currentPosition, currentPixel);
}