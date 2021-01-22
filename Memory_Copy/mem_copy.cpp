#include <celerity/celerity.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

void set_parameters(int argc, char *argv[]);

int seed = 1;
int mem_size = 7e8;

int main(int argc, char *argv[]) {
    celerity::runtime::init(&argc, &argv);
    set_parameters(argc, argv);

    std::vector<int> mem_A(mem_size);

    // initialize some random numbers for copy
    srand(seed);
    for (size_t i = 0; i < mem_size; i++)
        mem_A[i] = rand() % std::numeric_limits<int>::max();

    {
        celerity::buffer<int, 1> input_buf(mem_A.data(), cl::sycl::range<1>(mem_size));
        celerity::buffer<int, 1> output_buf((cl::sycl::range<1>(mem_size)));
        celerity::distr_queue queue;

        queue.submit([=](celerity::handler &cgh) {
            auto mem_input = input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
            auto mem_output = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
            // execute copy to ensure that the compiler doesn't optimize because of unused buffers
            cgh.parallel_for<class PrefetchKernel>(cl::sycl::range<1>(mem_size), [=](cl::sycl::item<1> item) {
                mem_output[item] = mem_input[item];
            });
        });

        // wait on host until prefetching is done
        queue.slow_full_sync();
        auto start = std::chrono::high_resolution_clock::now();
        queue.submit([=](celerity::handler &cgh) {
            auto mem_input = input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
            auto mem_output = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
            cgh.parallel_for<class MemCopyKernel>(cl::sycl::range<1>(mem_size), [=](cl::sycl::item<1> item) {
                mem_output[item] = mem_input[item];
            });
        });

        // wait on host until whole copy opertaion is done
        queue.slow_full_sync();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Bandwidth: " << ((sizeof(int) * mem_size) / 1e9) / (duration.count() / 1e6) * 2 << "GB/sec" << std::endl;
        std::cout << "Time:" << duration.count() << std::endl;
    }

    return EXIT_SUCCESS;
}

void set_parameters(int argc, char *argv[]) {
    bool is_numeric = true;
    if (argc >= 2) {
        for (size_t i = 1; i < argc; i++) {
            is_numeric = true;
            if (!strcmp(argv[i], "-s")) {
                i++;
                if (i < argc) {
                    for (size_t j = 0; j < strlen(argv[i]); j++)
                        if (!isdigit(argv[i][j]))
                            is_numeric = false;
                    if (is_numeric)
                        seed = atoi(argv[i]);
                }
            } else if (!strcmp(argv[i], "-m")) {
                i++;
                if (i < argc) {
                    for (size_t j = 0; j < strlen(argv[i]); j++)
                        if (!isdigit(argv[i][j]))
                            is_numeric = false;
                    if (is_numeric)
                        mem_size = atoi(argv[i]);
                }
            }
        }
    }
}
