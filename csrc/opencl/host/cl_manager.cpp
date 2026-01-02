#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// Use generic OpenCL header
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class CLManager {
public:
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program program;

    CLManager() {
        context = nullptr;
        queue = nullptr;
        program = nullptr;
    }

    ~CLManager() {
        if (queue) clReleaseCommandQueue(queue);
        if (program) clReleaseProgram(program);
        if (context) clReleaseContext(context);
    }

    // 1. Initialize: Find GPU and create context
    bool init() {
        cl_int err;
        cl_platform_id platform;
        cl_uint num_platforms;

        // Get Platform
        err = clGetPlatformIDs(1, &platform, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            std::cerr << "[OpenCL] Error: No platforms found." << std::endl;
            return false;
        }

        // Get Device (GPU preferred)
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Warning: No GPU found, trying CPU..." << std::endl;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        }
        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL] Error: No devices found." << std::endl;
            return false;
        }

        // Create Context
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;

        // Create Command Queue
        // Note: clCreateCommandQueue is deprecated in OpenCL 2.0, using properties preferred
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) return false;

        char name[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
        std::cout << "[OpenCL] Initialized on: " << name << std::endl;
        return true;
    }

    // 2. Load and Build Kernel Source
    bool load_kernels(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[OpenCL] Error: Could not open kernel file: " << filepath << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string src = buffer.str();
        const char* src_ptr = src.c_str();

        cl_int err;
        program = clCreateProgramWithSource(context, 1, &src_ptr, NULL, &err);
        
        // Build
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
            std::cerr << "[OpenCL] Build Error:\n" << log.data() << std::endl;
            return false;
        }
        return true;
    }

    // 3. Helper to Create Buffers
    cl_mem create_buffer(size_t size, void* host_ptr, bool read_only) {
        cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
        if (host_ptr) {
            flags |= CL_MEM_COPY_HOST_PTR;
            return clCreateBuffer(context, flags, size, host_ptr, NULL);
        } else {
            return clCreateBuffer(context, flags, size, NULL, NULL);
        }
    }
};

// Optional: A simple main to test independent of Python
int main() {
    CLManager cl;
    if (cl.init()) {
        std::cout << "[OpenCL] Host Manager Ready." << std::endl;
    }
    return 0;
}