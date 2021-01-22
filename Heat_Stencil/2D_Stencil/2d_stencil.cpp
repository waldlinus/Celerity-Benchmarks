#include <celerity/celerity.h>
#include <ctype.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

typedef float value_t;

bool check_parameters(int argc, char *argv[]);
void compute_sequential(std::vector<value_t> &result, int height, int width, int timesteps, int heat_source_x, int heat_source_y, value_t min, value_t max);
void print_matrix(const std::vector<value_t> &arr, int height, int width);
void printTemperature(const std::vector<value_t> &r, int N, int M, value_t min, value_t max);

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
            std::cout << "expected input: ./[file_name] timesteps height width [optional parameters]" << std::endl;
        return EXIT_FAILURE;
    }

    const value_t max = 273 + 60;
    const value_t min = 273;
    const int timesteps = atoi(argv[1]);
    const int height = atoi(argv[2]);
    const int width = atoi(argv[3]);
    const int heat_source_x = width / 2;
    const int heat_source_y = height / 2;
    bool verification_success = true;
    std::vector<value_t> stencil(height * width);

    // initialize stencil
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            if (i == heat_source_y && j == heat_source_x)
                stencil[i * width + j] = 273 + 60;
            else
                stencil[i * width + j] = 273;
        }
    }

    {
        celerity::buffer<value_t, 2> input_buf(stencil.data(), cl::sycl::range<2>(height, width));
        celerity::buffer<value_t, 2> output_buf(cl::sycl::range<2>(height, width));
        celerity::distr_queue queue;

        MPI_Barrier(MPI_COMM_WORLD);
        celerity::experimental::bench::begin("main program");

        for (size_t i = 0; i < timesteps; i++) {
            queue.submit([=](celerity::handler &cgh) {
                auto stencil_input = input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
                auto stencil_output = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
                cgh.parallel_for<class HeatStencilKernel>(cl::sycl::range<2>(height, width), [=](cl::sycl::item<2> item) {
                    // heat source stays at the same temperature
                    if (item[0] == heat_source_y && item[1] == heat_source_x) {
                        stencil_output[item] = stencil_input[item];
                    } else {
                        // get current temperature
                        value_t tc = stencil_input[item];

                        // get temperatures left/right and up/down
                        value_t tl = (item[1] != 0) ? stencil_input[{item[0], item[1] - 1}] : tc;
                        value_t tr = (item[1] != width - 1) ? stencil_input[{item[0], item[1] + 1}] : tc;
                        value_t tu = (item[0] != 0) ? stencil_input[{item[0] - 1, item[1]}] : tc;
                        value_t td = (item[0] != height - 1) ? stencil_input[{item[0] + 1, item[1]}] : tc;

                        // update temperature at current position
                        stencil_output[item] = tc + 0.2f * (tl + tr + tu + td + (-4.0f * tc));
                    }
                });
            });

            if (draw_inter) {
                if ((i % 100) == 0 || i == (timesteps - 1)) {
                    queue.slow_full_sync();
                    queue.submit([=](celerity::handler &cgh) {
                        auto inter = output_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
                        cgh.host_task(celerity::on_master_node, [=]() {
                            std::vector<value_t> step(height * width);
                            for (size_t j = 0; j < height; j++)
                                for (size_t k = 0; k < width; k++)
                                    step[j * width + k] = inter[{j, k}];
                            std::cout << "Step: " << i << std::endl;
                            printTemperature(step, height, width, min, max);
                        });
                    });
                }
            }

            // swap buffers for the following time step
            celerity::buffer<value_t, 2> temp = input_buf;
            input_buf = output_buf;
            output_buf = temp;
        }

        // final swap to obtain result
        output_buf = input_buf;

        queue.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {
            auto result = output_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
            cgh.host_task(celerity::on_master_node, [=, &verification_success]() {
                celerity::experimental::bench::end("main program");

                if (sequential_check) {
                    std::vector<value_t> sequential(height * width);
                    std::cout << "Computing sequentially ..." << std::endl;
                    auto start = std::chrono::high_resolution_clock::now();
                    compute_sequential(sequential, height, width, timesteps, heat_source_x, heat_source_y, min, max);
                    auto end = std::chrono::high_resolution_clock::now();
                    for (size_t i = 0; i < height; i++) {
                        for (size_t j = 0; j < width; j++) {
                            value_t diff = result[{i, j}] - sequential[i * width + j];
                            if (diff > 10E-3 || diff < -10E-3) {
                                std::cout << "Error at position (" << i << ", " << j << "), expected value " << sequential[i * width + j] << ", actual value " << result[{i, j}] << std::endl;
                                verification_success = false;
                                break;
                            }
                        }
                        if (!verification_success)
                            break;
                    }
                    if (print_result && verification_success) {
                        std::cout << "result:" << std::endl;
                        for (size_t i = 0; i < height; i++) {
                            for (size_t j = 0; j < width; j++) {
                                std::cout << result[{i, j}] << " ";
                            }
                            std::cout << std::endl;
                        }
                        std::cout << std::endl;
                        std::cout << "sequential: " << std::endl;
                        print_matrix(sequential, height, width);
                    }
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Time sequential: " << duration.count() / 1e6 << "sec" << std::endl;
                } else {
                    for (size_t i = 0; i < height; i++) {
                        for (size_t j = 0; j < width; j++) {
                            if (result[{i, j}] < min || result[{i, j}] > max) {
                                std::cout << "Error at position (" << i << ", " << j << "), value: " << result[{i, j}] << std::endl;
                                verification_success = false;
                                break;
                            }
                        }
                        if (!verification_success)
                            break;
                    }

                    if (print_result && verification_success) {
                        for (size_t i = 0; i < height; i++) {
                            for (size_t j = 0; j < width; j++) {
                                std::cout << result[{i, j}] << " ";
                            }
                            std::cout << std::endl;
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
    if (argc < 4) {
        return false;
    }

    bool is_numeric = true;
    for (size_t i = 1; i < 4; i++) {
        for (size_t j = 0; j < strlen(argv[i]); j++)
            if (!isdigit(argv[i][j]))
                is_numeric = false;
    }

    if (argc >= 5) {
        for (size_t i = 4; i < argc; i++) {
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

void compute_sequential(std::vector<value_t> &result, int height, int width, int timesteps, int heat_source_x, int heat_source_y, value_t min, value_t max) {
    std::vector<value_t> temp_vec(height * width);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            if (i == heat_source_y && j == heat_source_x)
                result[i * width + j] = max;
            else
                result[i * width + j] = min;
        }
    }

    for (size_t t = 0; t < timesteps; t++) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                // heat source stays at the same temperature
                if (j == heat_source_x && i == heat_source_y) {
                    temp_vec[i * width + j] = result[i * width + j];
                    continue;
                }

                // get current temperature at (i,j)
                value_t tc = result[i * width + j];

                // get temperatures left/right and up/down
                value_t tl = (j != 0) ? result[i * width + (j - 1)] : tc;
                value_t tr = (j != width - 1) ? result[i * width + (j + 1)] : tc;
                value_t tu = (i != 0) ? result[(i - 1) * width + j] : tc;
                value_t td = (i != height - 1) ? result[(i + 1) * width + j] : tc;

                // update temperature at current point
                temp_vec[i * width + j] = tc + 0.2 * (tl + tr + tu + td + (-4.0f * tc));
            }
        }

        // swap buffers (just pointers, not content)
        result.swap(temp_vec);
        if (print_progression && (t % 100 == 0 || t == timesteps - 1))
            std::cout << "\rProgression: " << (int)(((float)t) / (timesteps - 1) * 100) << "%" << std::flush;
    }
    if (print_progression)
        std::cout << std::endl;
}

void print_matrix(const std::vector<value_t> &arr, int height, int width) {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            if (j == width)
                std::cout << arr[i * width + j];
            else
                std::cout << arr[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printTemperature(const std::vector<value_t> &r, int N, int M, value_t min, value_t max) {
    const char *colors = " .-:=+*#%@";
    const int numColors = 10;

    // set the 'render' resolution
    int H = 30;
    int W = 50;

    // step size in each dimension
    int sH = N / H;
    int sW = M / W;

    // upper wall
    for (int i = 0; i < W + 2; i++) {
        std::cout << "X";
    }
    std::cout << std::endl;

    // room
    for (int i = 0; i < H; i++) {
        // left wall
        std::cout << "X";

        // actual room
        for (int j = 0; j < W; j++) {
            // get max temperature in this tile
            value_t max_t = 0;
            for (int x = sH * i; x < sH * i + sH; x++) {
                for (int y = sW * j; y < sW * j + sW; y++) {
                    max_t = (max_t < r[x * M + y]) ? r[x * M + y] : max_t;
                }
            }
            value_t temp = max_t;

            // pick the 'color'
            int c = ((temp - min) / (max - min)) * numColors;
            c = (c >= numColors) ? numColors - 1 : ((c < 0) ? 0 : c);

            // print the average temperature
            std::cout << colors[c];
        }

        // right wall
        std::cout << "X" << std::endl;
    }

    // lower wall
    for (int i = 0; i < W + 2; i++) {
        std::cout << "X";
    }
    std::cout << std::endl;
}
