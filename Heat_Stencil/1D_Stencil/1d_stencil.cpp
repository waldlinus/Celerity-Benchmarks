#include <celerity/celerity.h>
#include <ctype.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

typedef float value_t;

bool check_parameters(int argc, char *argv[]);
void compute_sequential(std::vector<value_t> &result, int length, int timesteps, int heat_source, value_t min, value_t max);
void print_array(const std::vector<value_t> &arr, int length);
void printTemperature(const std::vector<value_t> &r, int N, value_t min, value_t max);

bool sequential_check = false;
bool print_result = false;
bool print_progression = false;
bool draw_inter = false;

int main(int argc, char *argv[]) {
    celerity::runtime::init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool is_master = rank == 0;

    if (!check_parameters(argc, argv)) {
        if (is_master)
            std::cout << "Expected input: ./[file_name] timesteps length [optional parameters]" << std::endl;
        return EXIT_FAILURE;
    }

    const value_t max = 273 + 60;
    const value_t min = 273;
    const int timesteps = atoi(argv[1]);
    const int length = atoi(argv[2]);
    const int heat_source = length / 2;
    bool verification_success = true;
    std::vector<value_t> stencil(length);

    // initialize stencil
    for (size_t i = 0; i < length; i++) {
        if (i == heat_source)
            stencil[i] = max;
        else
            stencil[i] = min;
    }

    {
        celerity::buffer<value_t, 1> input_buf(stencil.data(), cl::sycl::range<1>(length));
        celerity::buffer<value_t, 1> output_buf((cl::sycl::range<1>(length)));
        celerity::distr_queue queue;

        MPI_Barrier(MPI_COMM_WORLD);
        celerity::experimental::bench::begin("main program");

        for (size_t i = 0; i < timesteps; i++) {
            queue.submit([=](celerity::handler &cgh) {
                auto stencil_input = input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<1>(1));
                auto stencil_output = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
                cgh.parallel_for<class HeatStencilKernel>(cl::sycl::range<1>(length), [=](cl::sycl::item<1> item) {
                    // heat source stays at the same temperature
                    if (item[0] == heat_source) {
                        stencil_output[item[0]] = stencil_input[item[0]];
                    } else {
                        // get current temperature
                        value_t tc = stencil_input[item[0]];

                        // get temperatures left and right
                        value_t tl = (item[0] != 0) ? stencil_input[item[0] - 1] : tc;
                        value_t tr = (item[0] != length - 1) ? stencil_input[item[0] + 1] : tc;

                        // update temperature at current position
                        stencil_output[item[0]] = tc + 0.2f * (tl + tr + (-2.0f * tc));
                    }
                });
            });

            if (draw_inter) {
                if ((i % 100) == 0 || i == (timesteps - 1)) {
                    queue.slow_full_sync();
                    queue.submit([=](celerity::handler &cgh) {
                        auto inter = output_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>());
                        cgh.host_task(celerity::on_master_node, [=]() {
                            std::vector<value_t> step(length);
                            for (size_t j = 0; j < length; j++)
                                step[j] = inter[j];
                            std::cout << "Step: " << i << std::endl;
                            printTemperature(step, length, min, max);
                        });
                    });
                }
            }

            // swap buffers for the following time step
            celerity::buffer<value_t, 1> temp = input_buf;
            input_buf = output_buf;
            output_buf = temp;
        }

        // final swap to obtain result
        output_buf = input_buf;

        queue.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {
            auto result = output_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>());
            cgh.host_task(celerity::on_master_node, [=, &verification_success]() {
                celerity::experimental::bench::end("main program");
                if (sequential_check) {
                    std::vector<value_t> sequential(length);
                    std::cout << "Computing sequentially ..." << std::endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    compute_sequential(sequential, length, timesteps, heat_source, min, max);
                    auto end = std::chrono::high_resolution_clock::now();
                    for (size_t i = 0; i < length; i++) {
                        value_t diff = result[i] - sequential[i];
                        if (diff > 10E-3 || diff < -10E-3) {
                            std::cout << "Error at position " << i << ", expected value " << sequential[i] << ", actual value " << result[i] << std::endl;
                            verification_success = false;
                            break;
                        }
                    }
                    if (print_result && verification_success) {
                        std::cout << "result: ";
                        for (size_t i = 0; i < length; i++) {
                            std::cout << result[i] << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "sequential: ";
                        print_array(sequential, length);
                    }
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Time sequential: " << duration.count() / 1e6 << "sec" << std::endl;
                } else {
                    for (size_t i = 0; i < length; i++) {
                        if (result[i] < min || result[i] > max) {
                            std::cout << "Error at position " << i << ", value: " << result[i] << std::endl;
                            verification_success = false;
                            break;
                        }
                    }

                    if (print_result && verification_success) {
                        for (size_t i = 0; i < length; i++) {
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
        for (size_t i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "-s"))
                sequential_check = true;
            else if (!strcmp(argv[i], "-p"))
                print_result = true;
            else if (!strcmp(argv[i], "-d"))
                draw_inter = true;
            else if (!strcmp(argv[i], "-pr"))
                print_progression = true;
        }
    }

    return is_numeric;
}

void compute_sequential(std::vector<value_t> &result, int length, int timesteps, int heat_source, value_t min, value_t max) {
    std::vector<value_t> temp_vec(length);

    for (size_t i = 0; i < length; i++) {
        if (i == heat_source)
            result[i] = max;
        else
            result[i] = min;
    }

    for (size_t t = 0; t < timesteps; t++) {
        for (size_t i = 0; i < length; i++) {
            // heat source stays at the same temperature
            if (i == heat_source) {
                temp_vec[i] = result[i];
                continue;
            }

            // get current temperature at (i,j)
            value_t tc = result[i];

            // get temperatures left/right and up/down
            value_t tl = (i != 0) ? result[i - 1] : tc;
            value_t tr = (i != length - 1) ? result[i + 1] : tc;

            // update temperature at current point
            temp_vec[i] = tc + 0.2 * (tl + tr + (-2.0f * tc));
        }

        // swap buffers (just pointers, not content)
        result.swap(temp_vec);
        if (print_progression && (t % 100 == 0 || t == timesteps - 1))
            std::cout << "\rProgression: " << (int)(((float)t) / (timesteps - 1) * 100) << "%" << std::flush;
    }
    if (print_progression)
        std::cout << std::endl;
}

void print_array(const std::vector<value_t> &arr, int length) {
    for (size_t i = 0; i < length; i++) {
        if (i == length)
            std::cout << arr[i];
        else
            std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void printTemperature(const std::vector<value_t> &r, int N, value_t min, value_t max) {
    const char *colors = " .-:=+*#%@";
    const int numColors = 10;

    // set the 'render' resolution
    int W = 50;

    // step size in each dimension
    int sW = N / W;

    // upper wall
    for (int i = 0; i < W + 2; i++) {
        std::cout << "X";
    }
    std::cout << std::endl;

    // left wall
    std::cout << "X";

    // room
    for (int i = 0; i < W; i++) {
        // actual room
        // get max temperature in this tile
        value_t max_t = 0;
        for (int x = sW * i; x < sW * i + sW; x++)
            max_t = (max_t < r[x]) ? r[x] : max_t;
        value_t temp = max_t;

        // pick the 'color'
        int c = ((temp - min) / (max - min)) * numColors;
        c = (c >= numColors) ? numColors - 1 : ((c < 0) ? 0 : c);

        // print the average temperature
        std::cout << colors[c];
    }
    // right wall
    std::cout << "X" << std::endl;

    // lower wall
    for (int i = 0; i < W + 2; i++) {
        std::cout << "X";
    }
    std::cout << std::endl
              << std::endl;
}
