#include <iostream>
#include <string>
#include <signal.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>

#include "Utils.h"

void PrintUsage()
{
    std::cout << "Usage: OpenCL-Test" << std::endl;
}

void SignalHandler(int signum)
{
}

void VectorAdditionTest()
{
    std::cout << "Running vector addition test..." << std::endl;

    const int LIST_SIZE = 1024 * 1024;
    std::cout << "List size: " << LIST_SIZE << std::endl;

    // local input buffers
    std::vector<int> A, B;
    A.reserve(LIST_SIZE);
    B.reserve(LIST_SIZE);

    // local output buffer
    std::vector<int> outputVector;
    outputVector.reserve(LIST_SIZE);

    // Initialize input
    for (int i = 0; i < LIST_SIZE; i++)
    {
        A[i] = i;
        B[i] = -i;
    }

    // Load kernel source code into array
    char *source_str = 0;
    size_t source_size = 0;
    const std::string kernelFile = "kernels/vector_addition.cl";
    if (!LoadKernelSource(kernelFile, &source_str, source_size))
    {
        return;
    }
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Number of CL platforms: %d\n", ret_num_platforms);

    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id, &ret_num_devices);
    printf("Number of CL devices: %d\n", ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    const size_t N_ITERATIONS = 10000;
    size_t i = 0;
    bool success = true;

    auto startTime = std::chrono::steady_clock::now();
    for (i = 0; i < N_ITERATIONS; i++)
    {
        // Copy the lists A and B to their respective memory buffers
        THROW_ON_FAIL(clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                LIST_SIZE * sizeof(int), A.data(), 0, NULL, NULL));
        THROW_ON_FAIL(clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
                LIST_SIZE * sizeof(int), B.data(), 0, NULL, NULL));

        // Enqueue the job
        THROW_ON_FAIL(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL));

        // Read the memory output buffer on the device to the local output variable
        THROW_ON_FAIL(clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0, 
                LIST_SIZE * sizeof(int), outputVector.data(), 0, NULL, NULL));

        // Check result
        size_t listSum = 0;
        for(size_t i = 0; i < LIST_SIZE; i++)
            listSum += outputVector[i];

        if (listSum != 0)
        {
            success = false;
            break;
        }
    }
    size_t elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTime).count();

    printf("Avg time: %lf us (%lf FPS)\n", (double)elapsedTime / (double)i, (double)i / ((double)elapsedTime / (double)1E6));
    std::cout << "Number of iterations: " << i << std::endl;

    success ? printf("Test passed\n") : printf("Test failed\n");

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(output_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    std::cout << "Done" << std::endl;
}

void ImageProcessingTest()
{
    std::cout << "Running image processing test..." << std::endl;

    // read input img
    cv::Mat inputImgBGR = cv::imread("data/test.jpg");

    // Convert to BGRA
    cv::Mat inputImg;
    cvtColor(inputImgBGR, inputImg, cv::COLOR_BGR2BGRA);
    size_t nelements = inputImg.rows * inputImg.cols * inputImg.channels();

    printf("Image type: %s\n", CVTypeToStr(inputImg.type()).c_str());
    printf("Image size: %d %d %d\n", inputImg.rows, inputImg.cols, inputImg.channels());

    // allocate output img
    std::vector<char> outputBuffer;
    outputBuffer.reserve(nelements);
    std::fill(outputBuffer.begin(), outputBuffer.end(), 0);

    // Load kernel source code into array
    char *sourceStr = 0;
    size_t sourceSize = 0;
    const std::string kernelFile = "kernels/image_morphology.cl";
    if (!LoadKernelSource(kernelFile, &sourceStr, sourceSize))
    {
        return;
    }
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Number of CL platforms: %d\n", ret_num_platforms);

    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id, &ret_num_devices);
    printf("Number of CL devices: %d\n", ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    // Create memory buffers on the device for each image
    cl_image_format format;
    format.image_channel_order = CL_BGRA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = inputImg.cols;
    desc.image_height = inputImg.rows;
    desc.image_depth = 1;
    desc.image_array_size = 1;
    desc.image_row_pitch = inputImg.cols;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    cl_mem input_mem_obj = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &ret);

    cl_mem output_mem_obj = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &ret);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char **)&sourceStr, (const size_t *)&sourceSize, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "image_process", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem_obj);

    // Copy input image to device
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {(size_t)inputImg.cols, (size_t)inputImg.rows, 1};

    printf("foo1\n");
    THROW_ON_FAIL(clEnqueueWriteImage(command_queue, input_mem_obj, CL_TRUE,
        origin, region, 0, 0, (void*)inputImg.data, 0, NULL, NULL));

    printf("foo2\n");

    // Enqueue the job
    size_t global_item_size = 1;
    size_t local_item_size = 1;
    THROW_ON_FAIL(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
        &global_item_size, &local_item_size, 0, NULL, NULL));

    printf("foo3\n");

    // Read the memory output buffer on the device to the local output variable
    THROW_ON_FAIL(clEnqueueReadImage(command_queue, output_mem_obj, CL_TRUE,
        origin, region, 0, 0, (void*)outputBuffer.data(), 0, NULL, NULL));

    cv::Mat outputImg(inputImg.rows, inputImg.cols, CV_8UC4, (unsigned char*)outputBuffer.data());

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(input_mem_obj);
    ret = clReleaseMemObject(output_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    cv::imshow("Image Processing Test", outputImg);
    cv::waitKey(0);
}

void RunTest(int testId)
{
    switch(testId)
    {
        case 0:
            VectorAdditionTest();
            break;
        case 1:
            ImageProcessingTest();
            break;
        default:
            VectorAdditionTest();
    }
}

int main(int argc, char** argv)
{
    // Handle ctl-c events
    signal(SIGINT, SignalHandler);

    if(HasArg(argc, argv, "-h"))
    {
        PrintUsage();
        return 0;
    }

    int testId = 0;
    if(HasArg(argc, argv, "-t"))
    {
        GetScalarArg(argc, argv, "-t", testId);
    }

    RunTest(testId);

    return 0;
}
