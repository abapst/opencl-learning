#pragma once

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#define MAX_SOURCE_SIZE (0x100000)

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define THROW_ON_FAIL(retcode, ...)               \
    if (retcode != CL_SUCCESS)                    \
    {                                             \
        printf("[%s:%s:%d] Error: %d\n", __FILENAME__, __func__, __LINE__, retcode);    \
        throw(std::exception(__VA_ARGS__));       \
    }

inline std::string CVTypeToStr(int type)
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

inline bool HasArg(int argc, char**argv, const std::string& query)
{
    for (int i = 0; i < argc; i++)
    {
        if(std::string(argv[i]).find(query) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

template <typename T>
inline T ConvertTo (const std::string &str)
{
    std::istringstream ss(str);
    T value;
    ss >> value;
    return value;
}

template <class T>
inline bool GetScalarArg(int argc, char** argv, const std::string& query, T& value, const T& defaultValue = 0)
{
    for (int i = 0; i < argc; i++)
    {
        if(std::string(argv[i]).find(query) != std::string::npos)
        {
            if ((i + 1) < argc)
            {
                value = ConvertTo<T>(argv[i+1]);
                return true;
            }
        }
    }

    value = defaultValue;
    return false;
}

inline bool LoadKernelSource(const std::string& kernelFile, char** source, size_t& sourceSize)
{
    FILE *fp;
    errno_t err = fopen_s(&fp, kernelFile.c_str(), "r");
    if (err)
    {
        std::cout << "Failed to load kernel " << kernelFile << std::endl;
        return false;
    }

    *source = (char*)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread(*source, 1, MAX_SOURCE_SIZE, fp);

    if (!sourceSize)
    {
        std::cout << "Did not read any bytes from " << kernelFile << std::endl;
        return false;
    }

    if (ferror(fp))
    {
        std::cout << "Something went wrong reading " << kernelFile << ": " << ferror(fp) << std::endl;
        return false;
    }

    if (fclose(fp))
    {
        std::cout << "Failed to close stream while reading kernel: " << kernelFile << std::endl;
        return false;
    }

    printf("Loaded %s: %llu bytes\n", kernelFile.c_str(), sourceSize);
    return true;
}