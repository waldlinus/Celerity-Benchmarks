#include <celerity/celerity.h>
#include <ctype.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

typedef float factor;

#define DIM2_H 4
#define DIM2_A 4
#define DIM2_E 2

void deta(std::vector<factor> &a, int na);
void deth(std::vector<factor> &h, int nr);
bool check_parameters(int argc, char *argv[]);
void compute_sequential(std::vector<factor> &result, const std::vector<factor> &a, const std::vector<factor> &h, int na, int nr);
void print_array(const std::vector<factor> &arr, int length);

bool sequential_check = false;
bool print_result = false;
bool print_progression = false;

int main(int argc, char *argv[]) {
    celerity::runtime::init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool is_master = rank == 0;

    if (!check_parameters(argc, argv)) {
        if (is_master)
            std::cout << "Expected input: ./[file_name] atoms reflections [optional parameters]" << std::endl;
        return EXIT_FAILURE;
    }

    const int na = atoi(argv[1]);
    const int nr = atoi(argv[2]);
    int seed = 1;
    bool verification_success = true;
    int size_a = DIM2_A * na;
    int size_h = DIM2_H * nr;
    int size_E = DIM2_E * nr;
    std::vector<factor> a(size_a);
    std::vector<factor> h(size_h);
    srand(seed);

    deta(a, na);
    deth(h, nr);

    float twopi;
    twopi = 6.28318584f;
    float f2 = 0.0f;
    for (int i = 0; i < na; i++)
        f2 += a[DIM2_A * i] * a[DIM2_A * i];
    f2 = 1.0f / sqrtf(f2);

    {
        celerity::buffer<factor, 1> input_a(a.data(), cl::sycl::range<1>(size_a));
        celerity::buffer<factor, 1> input_h(h.data(), cl::sycl::range<1>(size_h));
        celerity::buffer<factor, 1> output_buf((cl::sycl::range<1>(size_E)));
        celerity::distr_queue queue;

        MPI_Barrier(MPI_COMM_WORLD);
        celerity::experimental::bench::begin("main program");

        queue.submit([=](celerity::handler &cgh) {
            auto buf_a = input_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<1>());
            auto buf_h = input_h.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<1>());
            auto res_E = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
            cgh.parallel_for<class CrystalKernel>(cl::sycl::range<1>(2 * nr), [=](cl::sycl::item<1> item) {
                factor A = 0.0f;
                factor B = 0.0f;
                factor arg;
                for (size_t i = 0; i < na; i++) {
                    factor A1, B1;
                    if (item[0] & 1) {
                        arg = twopi * (buf_h[DIM2_H / 2 * (item[0] - 1) + 0] * buf_a[DIM2_A * i + 1] +
                                       buf_h[DIM2_H / 2 * (item[0] - 1) + 1] * buf_a[DIM2_A * i + 2] +
                                       buf_h[DIM2_H / 2 * (item[0] - 1) + 2] * buf_a[DIM2_A * i + 3]);

                        B1 = cl::sycl::sin(arg);
                        B += buf_a[DIM2_A * i] * B1;
                    } else {
                        arg = twopi * (buf_h[DIM2_H / 2 * item[0] + 0] * buf_a[DIM2_A * i + 1] +
                                       buf_h[DIM2_H / 2 * item[0] + 1] * buf_a[DIM2_A * i + 2] +
                                       buf_h[DIM2_H / 2 * item[0] + 2] * buf_a[DIM2_A * i + 3]);

                        A1 = cl::sycl::cos(arg);
                        A += buf_a[DIM2_A * i] * A1;
                    }
                }

                if (item[0] & 1)
                    res_E[item[0]] = B * f2;
                else
                    res_E[item[0]] = A * f2;
            });
        });

        queue.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {
            auto result = output_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>());
            cgh.host_task(celerity::on_master_node, [=, &verification_success]() {
                celerity::experimental::bench::end("main program");

                if (sequential_check) {
                    std::vector<factor> sequential(size_E);
                    std::cout << "Computing sequentially ..." << std::endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    compute_sequential(sequential, a, h, na, nr);
                    auto end = std::chrono::high_resolution_clock::now();

                    for (size_t i = 0; i < size_E; i++) {
                        factor diff = result[i] - sequential[i];
                        if (diff > 10E-3 || diff < -10E-3) {
                            std::cout << "Error at position " << i << ", expected value " << sequential[i] << ", actual value " << result[i] << std::endl;
                            verification_success = false;
                            break;
                        }
                    }
                    if (print_result && verification_success) {
                        std::cout << "result: ";
                        for (size_t i = 0; i < size_E; i++) {
                            std::cout << result[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "sequential: ";
                        print_array(sequential, size_E);
                    }
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Time sequential: " << duration.count() / 1e6 << "sec" << std::endl;

                } else {
                    if (print_result && verification_success) {
                        for (size_t i = 0; i < size_E; i++) {
                            std::cout << result[i] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                if (verification_success)
                    std::cout << "Verification OK" << std::endl;
                else
                    std::cout << "Verification FAILED" << std::endl;
            });
        });
    }

    return verification_success ? EXIT_SUCCESS : EXIT_FAILURE;
}

void deta(std::vector<factor> &a, int na) {
    for (int i = 0; i < na; i++) {
        if (i & 1)
            a[DIM2_A * i] = 6.0;
        else
            a[DIM2_A * i] = 7.0;
        for (int j = 1; j < DIM2_A; j++)
            a[DIM2_A * i + j] = (float)(rand() % RAND_MAX) / (float)RAND_MAX;
    }
}

void deth(std::vector<factor> &h, int nr) {
    const int hmax = 20;
    const int kmax = 30;
    const int lmax = 15;
    for (int i = 0; i < nr; i++) {
        h[DIM2_H * i + 0] = rintf(2 * hmax * (float)(rand() % RAND_MAX) / (float)RAND_MAX - hmax);
        h[DIM2_H * i + 1] = rintf(2 * kmax * (float)(rand() % RAND_MAX) / (float)RAND_MAX - kmax);
        h[DIM2_H * i + 2] = rintf(2 * lmax * (float)(rand() % RAND_MAX) / (float)RAND_MAX - lmax);
    }
}

bool check_parameters(int argc, char *argv[]) {
    if (argc < 3) {
        return false;
    }

    bool is_numeric = true;
    for (size_t i = 1; i < 3; i++) {
        for (size_t j = 0; j < strlen(argv[i]); j++)
            if (!isdigit(argv[i][j]))
                is_numeric = false;
    }

    if (argc >= 4) {
        for (size_t i = 0; i < argc; i++) {
            if (!strcmp(argv[i], "-s"))
                sequential_check = true;
            else if (!strcmp(argv[i], "-p"))
                print_result = true;
            else if (!strcmp(argv[i], "-pr"))
                print_progression = true;
        }
    }

    return is_numeric;
}

void compute_sequential(std::vector<factor> &result, const std::vector<factor> &a, const std::vector<factor> &h, int na, int nr) {
    factor A, B, twopi;
    twopi = 6.28318584f;

    float f2 = 0.0f;
    for (size_t i = 0; i < na; i++)
        f2 += a[DIM2_A * i] * a[DIM2_A * i];
    f2 = 1.0f / sqrtf(f2);

    for (size_t i = 0; i < nr; i++) {
        A = 0.0f;
        B = 0.0f;
        for (size_t j = 0; j < na; j++) {
            float A1, B1;
            float arg = twopi * (h[DIM2_H * i + 0] * a[DIM2_A * j + 1] +
                                 h[DIM2_H * i + 1] * a[DIM2_A * j + 2] +
                                 h[DIM2_H * i + 2] * a[DIM2_A * j + 3]);
            sincosf(arg, &B1, &A1);
            A += a[DIM2_A * j] * A1;
            B += a[DIM2_A * j] * B1;
        }
        result[DIM2_E * i] = A * f2;
        result[DIM2_E * i + 1] = B * f2;
        if (print_progression && ((i % 100) == 0 || i == (nr - 1)))
            std::cout << "\rProgression: " << (int)(((float)i) / (nr - 1) * 100) << "%" << std::flush;
    }
    std::cout << std::endl;
}

void print_array(const std::vector<factor> &arr, int length) {
    for (size_t i = 0; i < length; i++) {
        if (i == length)
            std::cout << arr[i];
        else
            std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
