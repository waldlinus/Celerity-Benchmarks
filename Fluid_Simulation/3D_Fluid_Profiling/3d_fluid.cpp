#include <celerity/celerity.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

typedef float value_d;

#define LINEARSOLVERTIMES 10
#define IX(i, j, k) ((X + 2) * (Y + 2) * (i) + (X + 2) * (j) + (k))

// celerity functions
void clear_buffers(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z);

void velocity_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, celerity::buffer<value_d, 3> &u0, celerity::buffer<value_d, 3> &v0, celerity::buffer<value_d, 3> &w0, int X, int Y, int Z, value_d visc, value_d dt);

void density_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, int diff, value_d dt);

void add_source(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &s, int X, int Y, int Z, value_d dt);

void diffuse(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d diff, value_d dt);

void advect(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, int b, value_d dt);

void project(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, celerity::buffer<value_d, 3> &u0, celerity::buffer<value_d, 3> &v0, int X, int Y, int Z);

void lin_solve(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d a, value_d c);

void set_bnd(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b);

void set_bnd_faces(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b);

void set_bnd_edges(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b);

void set_bnd_corners(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b);

// sequential functions
void compute_seq(std::vector<value_d> &seq_u, std::vector<value_d> &seq_v, std::vector<value_d> &seq_w, std::vector<value_d> &seq_u_prev, std::vector<value_d> &seq_v_prev, std::vector<value_d> &seq_w_prev, std::vector<value_d> &seq_dens, std::vector<value_d> &seq_dens_prev, int size, int timesteps, int X, int Y, int Z, float dt, float diff, float visc, float force, float source);

void seq_get_force_source(std::vector<value_d> &d, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z);

void seq_vel_step(std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, std::vector<value_d> &u0, std::vector<value_d> &v0, std::vector<value_d> &w0, int X, int Y, int Z, float visc, float dt, double &time_add_source, double &time_diffuse, double &time_project, double &time_advect, double &time_lin_solve, double &time_set_bnd);

void seq_dens_step(std::vector<value_d> &x, std::vector<value_d> &x0, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, float diff, float dt, double &time_add_source, double &time_diffuse, double &time_advect, double &time_lin_solve, double &time_set_bnd);

void seq_add_source(std::vector<value_d> &x, std::vector<value_d> &s, int X, int Y, int Z, float dt);

void seq_diffuse(std::vector<value_d> &x, std::vector<value_d> &x0, int X, int Y, int Z, int b, float diff, float dt, double &time_lin_solve, double &time_set_bnd);

void seq_advect(std::vector<value_d> &d, std::vector<value_d> &d0, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, int b, float dt, double &time_set_bnd);

void seq_project(std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, std::vector<value_d> &p, std::vector<value_d> &div, int X, int Y, int Z, double &time_lin_solve, double &time_set_bnd);

void seq_lin_solve(std::vector<value_d> &x, std::vector<value_d> &x0, int X, int Y, int Z, int b, float a, float c, double &time_set_bnd);

void seq_set_bnd(std::vector<value_d> &x, int X, int Y, int Z, int b);

// helper functions
void print_array(const std::vector<value_d> &array, int X, int Y, int Z);

bool check_parameters(int argc, char *argv[]);

void print_help();

bool sequential_check = false;
bool print_result_dens = false;
bool print_seq_result_dens = false;

int main(int argc, char *argv[]) {
    celerity::runtime::init(&argc, &argv);

    if (!check_parameters(argc, argv)) {
        std::cout << "expected input: ./[file_name] timesteps height width depth [optional parameters]" << std::endl
                  << "type -h for help with optional parameters" << std::endl;
        return EXIT_FAILURE;
    }

    int timesteps = atoi(argv[1]);
    const int X = atoi(argv[2]);
    const int Y = atoi(argv[3]);
    const int Z = atoi(argv[4]);
    bool verification_success = true;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int size = (X + 2) * (Y + 2) * (Z + 2);
    value_d dt = 0.4f;        // time delta
    value_d diff = 0.0f;      // diffuse
    value_d visc = 0.0f;      // viscosity
    value_d source = 200.0f;  // density
    value_d force = 10.0f;

    std::vector<value_d> u(size);
    std::vector<value_d> v(size);
    std::vector<value_d> w(size);
    std::vector<value_d> u_prev(size);
    std::vector<value_d> v_prev(size);
    std::vector<value_d> w_prev(size);
    std::vector<value_d> dens(size);
    std::vector<value_d> dens_prev(size);

    {
        //put smoke in the center
        dens_prev[IX(X / 2, Y / 2, Z / 2)] = source;
        //add a force in v direction
        v_prev[IX(X / 2, 2, Z / 2)] = force * 10;

        celerity::buffer<value_d, 3> u_buf(u.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> v_buf(v.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> w_buf(w.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> u_prev_buf(u_prev.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> v_prev_buf(v_prev.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> w_prev_buf(w_prev.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> dens_buf(dens.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));
        celerity::buffer<value_d, 3> dens_prev_buf(dens_prev.data(), cl::sycl::range<3>(X + 2, Y + 2, Z + 2));

        celerity::distr_queue queue;

        MPI_Barrier(MPI_COMM_WORLD);
        celerity::experimental::bench::begin("main program");

        for (int i = 0; i < timesteps; i++) {
            queue.slow_full_sync();
            celerity::experimental::bench::begin("velocity step");
            velocity_step(queue, u_buf, v_buf, w_buf, u_prev_buf, v_prev_buf, w_prev_buf, X, Y, Z, visc, dt);
            queue.slow_full_sync();
            celerity::experimental::bench::end("velocity step");
            celerity::experimental::bench::begin("density step");
            density_step(queue, dens_buf, dens_prev_buf, u_buf, v_buf, w_buf, X, Y, Z, diff, dt);
            queue.slow_full_sync();
            celerity::experimental::bench::end("density step");
            celerity::experimental::bench::begin("clear buffers");
            clear_buffers(queue, dens_prev_buf, u_prev_buf, v_prev_buf, w_prev_buf, X, Y, Z);
            queue.slow_full_sync();
            celerity::experimental::bench::end("clear buffers");
        }

        queue.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {
            auto u_buffer = u_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto v_buffer = v_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto w_buffer = w_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto u_prev_buffer = u_prev_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto v_prev_buffer = v_prev_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto w_prev_buffer = w_prev_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto dens_buffer = dens_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto dens_prev_buffer = dens_prev_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());

            cgh.host_task(celerity::on_master_node, [=, &verification_success]() {
                celerity::experimental::bench::end("main program");
                // check master
                if (sequential_check) {
                    std::vector<value_d> seq_u(size);
                    std::vector<value_d> seq_v(size);
                    std::vector<value_d> seq_w(size);
                    std::vector<value_d> seq_u_prev(size);
                    std::vector<value_d> seq_v_prev(size);
                    std::vector<value_d> seq_w_prev(size);
                    std::vector<value_d> seq_dens(size);
                    std::vector<value_d> seq_dens_prev(size);

                    auto start = std::chrono::high_resolution_clock::now();
                    compute_seq(seq_u, seq_v, seq_w, seq_u_prev, seq_v_prev, seq_w_prev, seq_dens, seq_dens_prev, size, timesteps, X, Y, Z, dt, diff, visc, force, source);
                    auto end = std::chrono::high_resolution_clock::now();

                    for (size_t i = 0; i < X + 2; i++) {
                        for (size_t j = 0; j < Y + 2; j++) {
                            for (size_t k = 0; k < Z + 2; k++) {
                                // value_d diff_u = u_buffer[{i, j, k}] - seq_u[IX(i, j, k)];
                                // value_d diff_v = v_buffer[{i, j, k}] - seq_v[IX(i, j, k)];
                                // value_d diff_w = w_buffer[{i, j, k}] - seq_w[IX(i, j, k)];
                                // value_d diff_u_prev = u_prev_buffer[{i, j, k}] - seq_u_prev[IX(i, j, k)];
                                // value_d diff_v_prev = v_prev_buffer[{i, j, k}] - seq_v_prev[IX(i, j, k)];
                                // value_d diff_w_prev = w_prev_buffer[{i, j, k}] - seq_w_prev[IX(i, j, k)];
                                value_d diff_dens = dens_buffer[{i, j, k}] - seq_dens[IX(i, j, k)];
                                // value_d diff_dens_prev = dens_prev_buffer[{i, j, k}] - seq_dens_prev[IX(i, j, k)];

                                /*
                                if (diff_u > 10E-1 || diff_u < -10E-1) {
                                    std::cout << "u-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_u[IX(i, j, k)] << ", actual value " << u_buffer[{i, j, k}] << ", difference: " << diff_u << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                if (diff_v > 10E-1 || diff_v < -10E-1) {
                                    std::cout << "v-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_v[IX(i, j, k)] << ", actual value " << v_buffer[{i, j, k}] << ", difference: " << diff_v << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                if (diff_w > 10E-1 || diff_w < -10E-1) {
                                    std::cout << "w-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_w[IX(i, j, k)] << ", actual value " << w_buffer[{i, j, k}] << ", difference: " << diff_w << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                if (diff_u_prev > 10E-1 || diff_u_prev < -10E-1) {
                                    std::cout << "u_prev-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_u_prev[IX(i, j, k)] << ", actual value " << u_prev_buffer[{i, j, k}] << ", difference: " << diff_u_prev << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                if (diff_v_prev > 10E-1 || diff_v_prev < -10E-1) {
                                    std::cout << "v_prev-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_v_prev[IX(i, j, k)] << ", actual value " << v_prev_buffer[{i, j, k}] << ", difference: " << diff_u_prev << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                if (diff_w_prev > 10E-1 || diff_w_prev < -10E-1) {
                                    std::cout << "w_prev-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_w_prev[IX(i, j, k)] << ", actual value " << w_prev_buffer[{i, j, k}] << ", difference: " << diff_w_prev << std::endl;
                                    verification_success = false;
                                    break;
                                }*/
                                if (diff_dens > 10E-6 || diff_dens < -10E-6) {
                                    std::cout << "dens-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_dens[IX(i, j, k)] << ", actual value " << dens_buffer[{i, j, k}] << ", difference: " << diff_dens << std::endl;
                                    verification_success = false;
                                    break;
                                }
                                /*
                                if (diff_dens_prev > 10E-1 || diff_dens_prev < -10E-1) {
                                    std::cout << "dens_prev-buffer: Error at position (" << i << ", " << j << ", " << k << "), expected value " << seq_dens_prev[IX(i, j, k)] << ", actual value " << dens_prev_buffer[{i, j, k}] << ", difference: " << diff_dens_prev << std::endl;
                                    verification_success = false;
                                    break;
                                }*/
                            }
                            if (!verification_success)
                                break;
                        }
                        if (!verification_success)
                            break;
                    }

                    if (print_seq_result_dens) {
                        std::cout << "Print sequential array dens:" << std::endl;
                        print_array(seq_dens, X, Y, Z);
                    }

                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Time sequential: " << duration.count() / 1e6 << "sec" << std::endl;
                    if (verification_success) {
                        std::cout << "Verification OK" << std::endl;
                    }
                }

                std::vector<value_d> result_dens(size);

                for (size_t i = 0; i < X + 2; i++) {
                    for (size_t j = 0; j < Y + 2; j++) {
                        for (size_t k = 0; k < Z + 2; k++) {
                            result_dens[IX(i, j, k)] = dens_buffer[{i, j, k}];
                        }
                    }
                }

                if (print_result_dens) {
                    std::cout << "Print arry dens:" << std::endl;
                    print_array(result_dens, X, Y, Z);
                }
            });
        });
    }

    return verification_success ? EXIT_SUCCESS : EXIT_FAILURE;
}

// celerity functions
void clear_buffers(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z) {
    queue.submit([=](celerity::handler &cgh) {
        auto d_prev = d.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        auto u_prev = u.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        auto v_prev = v.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        auto w_prev = w.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        cgh.parallel_for<class clearBuffersKernel>(cl::sycl::range<3>(X + 2, Y + 2, Z + 2), [=](cl::sycl::item<3> item) {
            d_prev[item] = 0;
            u_prev[item] = 0;
            v_prev[item] = 0;
            w_prev[item] = 0;
        });
    });
}

void velocity_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, celerity::buffer<value_d, 3> &u0, celerity::buffer<value_d, 3> &v0, celerity::buffer<value_d, 3> &w0, int X, int Y, int Z, value_d visc, value_d dt) {
    queue.slow_full_sync();
    celerity::experimental::bench::begin("add source1");
    add_source(queue, u, u0, X, Y, Z, dt);
    add_source(queue, v, v0, X, Y, Z, dt);
    add_source(queue, w, w0, X, Y, Z, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("add source1");
    std::swap(u0, u);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("diffuse1");
    diffuse(queue, u, u0, X, Y, Z, 1, visc, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("diffuse1");
    std::swap(v0, v);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("diffuse2");
    diffuse(queue, v, v0, X, Y, Z, 2, visc, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("diffuse2");
    std::swap(w0, w);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("diffuse3");
    diffuse(queue, w, w0, X, Y, Z, 3, visc, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("diffuse3");
    celerity::experimental::bench::begin("project1");
    project(queue, u, v, w, u0, v0, X, Y, Z);
    queue.slow_full_sync();
    celerity::experimental::bench::end("project1");
    std::swap(u0, u);
    std::swap(v0, v);
    std::swap(w0, w);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("advect1");
    advect(queue, u, u0, u0, v0, w0, X, Y, Z, 1, dt);
    advect(queue, v, v0, u0, v0, w0, X, Y, Z, 2, dt);
    advect(queue, w, w0, u0, v0, w0, X, Y, Z, 3, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("advect1");
    celerity::experimental::bench::begin("project2");
    project(queue, u, v, w, u0, v0, X, Y, Z);
    queue.slow_full_sync();
    celerity::experimental::bench::end("project2");
}

void density_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, int diff, value_d dt) {
    celerity::experimental::bench::begin("add source2");
    add_source(queue, x, x0, X, Y, Z, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("add source2");
    std::swap(x0, x);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("diffuse4");
    diffuse(queue, x, x0, X, Y, Z, 0, diff, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("diffuse4");
    std::swap(x0, x);
    queue.slow_full_sync();
    celerity::experimental::bench::begin("advect2");
    advect(queue, x, x0, u, v, w, X, Y, Z, 0, dt);
    queue.slow_full_sync();
    celerity::experimental::bench::end("advect2");
}

void add_source(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &s, int X, int Y, int Z, value_d dt) {
    queue.submit([=](celerity::handler &cgh) {
        auto density = x.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<3>());
        auto source = s.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());
        cgh.parallel_for<class addSource>(cl::sycl::range<3>(X + 2, Y + 2, Z + 2), [=](cl::sycl::item<3> item) {
            density[item] += dt * source[item];
        });
    });
}

void diffuse(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d diff, value_d dt) {
    int max = std::max(std::max(X, Y), std::max(Y, Z));
    value_d a = dt * diff * max * max * max;
    celerity::experimental::bench::begin("lin solve1");
    lin_solve(queue, x, x0, X, Y, Z, b, a, 1 + 6 * a);
    queue.slow_full_sync();
    celerity::experimental::bench::end("lin solve1");
}

void advect(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, int b, value_d dt) {
    queue.submit([=](celerity::handler &cgh) {
        auto density = x.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        auto prev_density = x0.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<3>());
        auto u_buffer = u.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());
        auto v_buffer = v.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());
        auto w_buffer = w.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());

        cgh.parallel_for<class AdvectKernel>(
            cl::sycl::range<3>(X, Y, Z),
            cl::sycl::id<3>(1, 1, 1),
            [=](cl::sycl::item<3> item) {
                value_d dtx = dt * std::max(std::max(X, Y), std::max(Y, Z));
                value_d dty = dt * std::max(std::max(X, Y), std::max(Y, Z));
                value_d dtz = dt * std::max(std::max(X, Y), std::max(Y, Z));

                value_d x = item[0] - dtx * u_buffer[item];
                value_d y = item[1] - dty * v_buffer[item];
                value_d z = item[2] - dtz * w_buffer[item];

                if (x < 0.5f)
                    x = 0.5f;
                if (x > X + 0.5f)
                    x = X + 0.5f;
                size_t i0 = (size_t)x;
                size_t i1 = i0 + 1;

                if (y < 0.5f)
                    y = 0.5f;
                if (y > Y + 0.5f)
                    y = Y + 0.5f;
                size_t j0 = (size_t)y;
                size_t j1 = j0 + 1;

                if (z < 0.5f)
                    z = 0.5f;
                if (z > Z + 0.5f)
                    z = Z + 0.5f;
                size_t k0 = (size_t)z;
                size_t k1 = k0 + 1;

                value_d s1 = x - i0;
                value_d s0 = 1 - s1;

                value_d t1 = y - j0;
                value_d t0 = 1 - t1;

                value_d u1 = z - k0;
                value_d u0 = 1 - u1;

                density[item] = s0 * (t0 * u0 * prev_density[{i0, j0, k0}] + t1 * u0 * prev_density[{i0, j1, k0}] + t0 * u1 * prev_density[{i0, j0, k1}] + t1 * u1 * prev_density[{i0, j1, k1}]) + s1 * (t0 * u0 * prev_density[{i1, j0, k0}] + t1 * u0 * prev_density[{i1, j1, k0}] + t0 * u1 * prev_density[{i1, j0, k1}] + t1 * u1 * prev_density[{i1, j1, k1}]);
            });
    });
    queue.slow_full_sync();
    celerity::experimental::bench::begin("set bnd1");
    set_bnd(queue, x, X, Y, Z, b);
    queue.slow_full_sync();
    celerity::experimental::bench::end("set bnd1");
}

void project(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, celerity::buffer<value_d, 3> &u0, celerity::buffer<value_d, 3> &v0, int X, int Y, int Z) {
    queue.submit([=](celerity::handler &cgh) {
        auto u_buffer = u.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        auto v_buffer = v.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        auto w_buffer = w.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        auto u0_buffer = u0.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());
        auto v0_buffer = v0.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<3>());

        cgh.parallel_for<class FirstProjectKernel>(cl::sycl::range<3>(X, Y, Z), cl::sycl::id<3>(1, 1, 1), [=](cl::sycl::item<3> item) {
            v0_buffer[item] = -1.0 / 3.0 * ((u_buffer[{item[0] + 1, item[1], item[2]}] - u_buffer[{item[0] - 1, item[1], item[2]}]) / X + (v_buffer[{item[0], item[1] + 1, item[2]}] - v_buffer[{item[0], item[1] - 1, item[2]}]) / X + (w_buffer[{item[0], item[1], item[2] + 1}] - w_buffer[{item[0], item[1], item[2] - 1}]) / X);
            u0_buffer[item] = 0;
        });
    });
    queue.slow_full_sync();
    celerity::experimental::bench::begin("set bnd2");
    set_bnd(queue, v0, X, Y, Z, 0);
    set_bnd(queue, u0, X, Y, Z, 0);
    queue.slow_full_sync();
    celerity::experimental::bench::end("set bnd2");

    celerity::experimental::bench::begin("lin solve2");
    lin_solve(queue, u0, v0, X, Y, Z, 0, 1, 6);
    queue.slow_full_sync();
    celerity::experimental::bench::end("lin solve2");

    queue.submit([=](celerity::handler &cgh) {
        auto u_buffer = u.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<3>());
        auto v_buffer = v.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<3>());
        auto w_buffer = w.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<3>());
        auto u0_buffer = u0.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<3>(1, 1, 1));

        cgh.parallel_for<class SecondProjectKernel>(cl::sycl::range<3>(X, Y, Z), cl::sycl::id<3>(1, 1, 1), [=](cl::sycl::item<3> item) {
            u_buffer[item] -= 0.5f * X * (u0_buffer[{item[0] + 1, item[1], item[2]}] - u0_buffer[{item[0] - 1, item[1], item[2]}]);
            v_buffer[item] -= 0.5f * X * (u0_buffer[{item[0], item[1] + 1, item[2]}] - u0_buffer[{item[0], item[1] - 1, item[2]}]);
            w_buffer[item] -= 0.5f * X * (u0_buffer[{item[0], item[1], item[2] + 1}] - u0_buffer[{item[0], item[1], item[2] - 1}]);
        });
    });
    queue.slow_full_sync();
    celerity::experimental::bench::begin("set bnd3");
    set_bnd(queue, u, X, Y, Z, 1);
    set_bnd(queue, v, X, Y, Z, 2);
    set_bnd(queue, w, X, Y, Z, 3);
    queue.slow_full_sync();
    celerity::experimental::bench::end("set bnd3");
}

/*
void lin_solve(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d a, value_d c) {
    size_t l;
    // iterate the solver
    for (l = 0; l < LINEARSOLVERTIMES; l++) {
        queue.submit(celerity::allow_by_ref, [&, X, Y, Z, a, c](celerity::handler &cgh) {
            auto dens = x.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
            auto dens_prev = x0.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());

            cgh.host_task(celerity::on_master_node, [=]() {
                // update for each cell
                for (size_t i = 1; i <= X; i++) {
                    for (size_t j = 1; j <= Y; j++) {
                        for (size_t k = 1; k <= Z; k++) {
                            dens[{i, j, k}] = (dens_prev[{i, j, k}] + a * (dens[{i - 1, j, k}] + dens[{i + 1, j, k}] + dens[{i, j - 1, k}] + dens[{i, j + 1, k}] + dens[{i, j, k - 1}] + dens[{i, j, k + 1}])) / c;
                        }
                    }
                }
            });
        });
        queue.slow_full_sync();
        celerity::experimental::bench::begin("set bnd4");
        set_bnd(queue, x, X, Y, Z, b);
        queue.slow_full_sync();
        celerity::experimental::bench::end("set bnd4");
    }
}
*/

void lin_solve(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d a, value_d c) {
    for (size_t i = 0; i < LINEARSOLVERTIMES; i++) {
        queue.submit([=](celerity::handler &cgh) {
            auto dens = x.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
            auto dens_prev = x0.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());
            cgh.parallel_for<class linSolve>(cl::sycl::range<3>(X, Y, Z), cl::sycl::id<3>(1, 1, 1), [=](cl::sycl::item<3> item) {
                dens[item] = (dens_prev[item] + a * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}] + dens[{item[0], item[1], item[2] + 1}])) / c;
            });
        });
        queue.slow_full_sync();
        celerity::experimental::bench::begin("set bnd4");
        set_bnd(queue, x, X, Y, Z, b);
        queue.slow_full_sync();
        celerity::experimental::bench::end("set bnd4");
    }
}

void set_bnd(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b) {
    set_bnd_faces(queue, x, X, Y, Z, b);
    set_bnd_edges(queue, x, X, Y, Z, b);
    set_bnd_corners(queue, x, X, Y, Z, b);
    /*
    queue.submit(celerity::allow_by_ref, [&, X, Y, Z, b](celerity::handler &cgh) {
        auto dens = x.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        cgh.host_task(celerity::on_master_node, [=]() {
            size_t Xs = X;
            size_t Ys = Y;
            size_t Zs = Z;

            // Setting faces
            for (size_t i = 1; i <= Xs; i++) {
                for (size_t j = 1; j <= Ys; j++) {
                    dens[{i, j, 0}] = b == 3 ? -dens[{i, j, 1}] : dens[{i, j, 1}];
                    dens[{i, j, Zs + 1}] = b == 3 ? -dens[{i, j, Zs}] : dens[{i, j, Zs}];
                }
            }

            for (size_t i = 1; i <= Ys; i++) {
                for (size_t j = 1; j <= Zs; j++) {
                    dens[{0, i, j}] = b == 1 ? -dens[{1, i, j}] : dens[{1, i, j}];
                    dens[{Xs + 1, i, j}] = b == 1 ? -dens[{Xs, i, j}] : dens[{Xs, i, j}];
                }
            }

            for (size_t i = 1; i <= Xs; i++) {
                for (size_t j = 1; j <= Zs; j++) {
                    dens[{i, 0, j}] = b == 2 ? -dens[{i, 1, j}] : dens[{i, 1, j}];
                    dens[{i, Ys + 1, j}] = b == 2 ? -dens[{i, Ys, j}] : dens[{i, Ys, j}];
                }
            }

            // Setting edges
            for (size_t i = 1; i <= Xs; i++) {
                dens[{i, 0, 0}] = 1.0 / 2.0 * (dens[{i, 1, 0}] + dens[{i, 0, 1}]);
                dens[{i, Ys + 1, 0}] = 1.0 / 2.0 * (dens[{i, Ys, 0}] + dens[{i, Ys + 1, 1}]);
                dens[{i, 0, Zs + 1}] = 1.0 / 2.0 * (dens[{i, 0, Zs}] + dens[{i, 1, Zs + 1}]);
                dens[{i, Ys + 1, Zs + 1}] = 1.0 / 2.0 * (dens[{i, Ys, Zs + 1}] + dens[{i, Ys + 1, Zs}]);
            }

            for (size_t i = 1; i <= Ys; i++) {
                dens[{0, i, 0}] = 1.0 / 2.0 * (dens[{1, i, 0}] + dens[{0, i, 1}]);
                dens[{Xs + 1, i, 0}] = 1.0 / 2.0 * (dens[{Xs, i, 0}] + dens[{Xs + 1, i, 1}]);
                dens[{0, i, Zs + 1}] = 1.0 / 2.0 * (dens[{0, i, Zs}] + dens[{1, i, Zs + 1}]);
                dens[{Xs + 1, i, Zs + 1}] = 1.0 / 2.0 * (dens[{Xs, i, Zs + 1}] + dens[{Xs + 1, i, Zs}]);
            }

            for (size_t i = 1; i <= Zs; i++) {
                dens[{0, 0, i}] = 1.0 / 2.0 * (dens[{0, 1, i}] + dens[{1, 0, i}]);
                dens[{0, Ys + 1, i}] = 1.0 / 2.0 * (dens[{0, Ys, i}] + dens[{1, Ys + 1, i}]);
                dens[{Xs + 1, 0, i}] = 1.0 / 2.0 * (dens[{Xs, 0, i}] + dens[{Xs + 1, 1, i}]);
                dens[{Xs + 1, Ys + 1, i}] = 1.0 / 2.0 * (dens[{Xs + 1, Ys, i}] + dens[{Xs, Ys + 1, i}]);
            }

            // Setting corners
            dens[{0, 0, 0}] = 1.0 / 3.0 * (dens[{1, 0, 0}] + dens[{0, 1, 0}] + dens[{0, 0, 1}]);
            dens[{0, Ys + 1, 0}] = 1.0 / 3.0 * (dens[{1, Ys + 1, 0}] + dens[{0, Ys, 0}] + dens[{0, Ys + 1, 1}]);

            dens[{Xs + 1, 0, 0}] = 1.0 / 3.0 * (dens[{Xs, 0, 0}] + dens[{Xs + 1, 1, 0}] + dens[{Xs + 1, 0, 1}]);
            dens[{Xs + 1, Ys + 1, 0}] = 1.0 / 3.0 * (dens[{Xs, Ys + 1, 0}] + dens[{Xs + 1, Ys, 0}] + dens[{Xs + 1, Ys + 1, 1}]);

            dens[{0, 0, Zs + 1}] = 1.0 / 3.0 * (dens[{1, 0, Zs + 1}] + dens[{0, 1, Zs + 1}] + dens[{0, 0, Zs}]);
            dens[{0, Ys + 1, Zs + 1}] = 1.0 / 3.0 * (dens[{1, Ys + 1, Zs + 1}] + dens[{0, Ys, Zs + 1}] + dens[{0, Ys + 1, Zs}]);

            dens[{Xs + 1, 0, Zs + 1}] = 1.0 / 3.0 * (dens[{Xs, 0, Zs + 1}] + dens[{Xs + 1, 1, Zs + 1}] + dens[{Xs + 1, 0, Zs}]);
            dens[{Xs + 1, Ys + 1, Zs + 1}] = 1.0 / 3.0 * (dens[{Xs, Ys + 1, Zs + 1}] + dens[{Xs + 1, Ys, Zs + 1}] + dens[{Xs + 1, Ys + 1, Zs}]);
        });
    });*/
}

void set_bnd_faces(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b) {
    queue.submit([=](celerity::handler &cgh) {
        auto dens = x.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        cgh.parallel_for<class setBoundariesFaces>(cl::sycl::range<3>(X + 2, Y + 2, Z + 2), [=](cl::sycl::item<3> item) {
            if (item[0] == 0 && item[1] != 0 && item[1] != Y + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = b == 1 ? -dens[{item[0] + 1, item[1], item[2]}] : dens[{item[0] + 1, item[1], item[2]}];

            if (item[0] == X + 1 && item[1] != 0 && item[1] != Y + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = b == 1 ? -dens[{item[0] - 1, item[1], item[2]}] : dens[{item[0] - 1, item[1], item[2]}];

            if (item[1] == 0 && item[0] != 0 && item[0] != X + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = b == 2 ? -dens[{item[0], item[1] + 1, item[2]}] : dens[{item[0], item[1] + 1, item[2]}];

            if (item[1] == Y + 1 && item[0] != 0 && item[0] != X + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = b == 2 ? -dens[{item[0], item[1] - 1, item[2]}] : dens[{item[0], item[1] - 1, item[2]}];

            if (item[2] == 0 && item[0] != 0 && item[0] != X + 1 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = b == 3 ? -dens[{item[0], item[1], item[2] + 1}] : dens[{item[0], item[1], item[2] + 1}];

            if (item[2] == Z + 1 && item[0] != 0 && item[0] != X + 1 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = b == 3 ? -dens[{item[0], item[1], item[2] - 1}] : dens[{item[0], item[1], item[2] - 1}];
        });
    });
}

void set_bnd_edges(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b) {
    queue.submit([=](celerity::handler &cgh) {
        auto dens = x.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        cgh.parallel_for<class setBoundariesEdges>(cl::sycl::range<3>(X + 2, Y + 2, Z + 2), [=](cl::sycl::item<3> item) {
            if (item[1] == 0 && item[2] == 0 && item[0] != 0 && item[0] != X + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[1] == Y + 1 && item[2] == 0 && item[0] != 0 && item[0] != X + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[1] == 0 && item[2] == Z + 1 && item[0] != 0 && item[0] != X + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[1] == Y + 1 && item[2] == Z + 1 && item[0] != 0 && item[0] != X + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == 0 && item[2] == 0 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == X + 1 && item[2] == 0 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == 0 && item[2] == Z + 1 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == X + 1 && item[2] == Z + 1 && item[1] != 0 && item[1] != Y + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == 0 && item[1] == 0 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}]);

            if (item[0] == X + 1 && item[1] == 0 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}]);

            if (item[0] == 0 && item[1] == Y + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}]);

            if (item[0] == X + 1 && item[1] == Y + 1 && item[2] != 0 && item[2] != Z + 1)
                dens[item] = 1.0 / 2.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}]);
        });
    });
}

void set_bnd_corners(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b) {
    queue.submit([=](celerity::handler &cgh) {
        auto dens = x.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
        cgh.parallel_for<class setBoundariesCorners>(cl::sycl::range<3>(X + 2, Y + 2, Z + 2), [=](cl::sycl::item<3> item) {
            if (item[0] == 0 && item[1] == 0 && item[2] == 0)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == 0 && item[1] == Y + 1 && item[2] == 0)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == X + 1 && item[1] == 0 && item[2] == 0)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == X + 1 && item[1] == Y + 1 && item[2] == 0)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] + 1}]);

            if (item[0] == 0 && item[1] == 0 && item[2] == Z + 1)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == 0 && item[1] == Y + 1 && item[2] == Z + 1)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == X + 1 && item[1] == 0 && item[2] == Z + 1)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);

            if (item[0] == X + 1 && item[1] == Y + 1 && item[2] == Z + 1)
                dens[item] = 1.0 / 3.0 * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}]);
        });
    });
}

// sequential functions
void compute_seq(std::vector<value_d> &seq_u, std::vector<value_d> &seq_v, std::vector<value_d> &seq_w, std::vector<value_d> &seq_u_prev, std::vector<value_d> &seq_v_prev, std::vector<value_d> &seq_w_prev, std::vector<value_d> &seq_dens, std::vector<value_d> &seq_dens_prev, int size, int timesteps, int X, int Y, int Z, float dt, float diff, float visc, float force, float source) {
    std::cout << "Sequential computation running ..." << std::endl;

    bool seq_add_source = true;
    bool seq_add_force_x = false;
    bool seq_add_force_y = true;
    bool seq_add_force_z = false;

    double time_force_source = 0.0;
    double time_vel_step = 0.0;
    double time_dens_step = 0.0;

    double time_add_source = 0.0;
    double time_diffuse = 0.0;
    double time_project = 0.0;
    double time_advect = 0.0;
    double time_lin_solve = 0.0;
    double time_set_bnd = 0.0;

    for (int i = 0; i < timesteps; i++) {
        auto time1 = std::chrono::high_resolution_clock::now();
        seq_get_force_source(seq_dens_prev, seq_u_prev, seq_v_prev, seq_w_prev, X, Y, Z, force, source, seq_add_source, seq_add_force_x, seq_add_force_y, seq_add_force_z);
        auto time2 = std::chrono::high_resolution_clock::now();
        seq_vel_step(seq_u, seq_v, seq_w, seq_u_prev, seq_v_prev, seq_w_prev, X, Y, Z, visc, dt, time_add_source, time_diffuse, time_project, time_advect, time_lin_solve, time_set_bnd);
        auto time3 = std::chrono::high_resolution_clock::now();
        seq_dens_step(seq_dens, seq_dens_prev, seq_u, seq_v, seq_w, X, Y, Z, diff, dt, time_add_source, time_diffuse, time_advect, time_lin_solve, time_set_bnd);
        auto time4 = std::chrono::high_resolution_clock::now();
        time_force_source += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
        time_vel_step += std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
        time_dens_step += std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();

        if (!(i % 10))
            std::cout << "Timesteps done: " << i << std::endl;
    }

    std::cout << "Sequential time summary:" << std::endl;
    std::cout << "Time seq get force source:\t" << time_force_source << std::endl;
    std::cout << "Time seq velocity step:\t" << time_vel_step << std::endl;
    std::cout << "Time seq density step:\t" << time_dens_step << std::endl;
    std::cout << "Time seq add source:\t" << time_add_source << std::endl;
    std::cout << "Time seq diffuse:\t" << time_diffuse << std::endl;
    std::cout << "Time seq project:\t" << time_project << std::endl;
    std::cout << "Time seq advect:\t" << time_advect << std::endl;
    std::cout << "Time seq lin solve:\t" << time_lin_solve << std::endl;
    std::cout << "Time seq set bnd:\t" << time_set_bnd << std::endl;
}

void seq_get_force_source(std::vector<value_d> &d, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z) {
    int size = (X + 2) * (Y + 2) * (Z + 2);

    for (int i = 0; i < size; i++) {
        u[i] = v[i] = w[i] = d[i] = 0.0f;
    }

    if (add_force_x) {
        int i = 2;
        int j = Y / 2;
        int k = Z / 2;

        if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z)
            return;
        u[IX(i, j, k)] = force * 10;
        add_force_x = false;
    }

    if (add_force_y) {
        int i = X / 2;
        int j = 2;
        int k = Z / 2;

        if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z)
            return;
        v[IX(i, j, k)] = force * 10;
        add_force_y = false;
    }

    if (add_force_z) {
        int i = X / 2;
        int j = Y / 2;
        int k = 2;

        if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z)
            return;
        w[IX(i, j, k)] = force * 10;
        add_force_z = false;
    }

    if (add_source) {
        int i = X / 2;
        int j = Y / 2;
        int k = Z / 2;
        d[IX(i, j, k)] = source;
        add_source = false;
    }
    return;
}

void seq_vel_step(std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, std::vector<value_d> &u0, std::vector<value_d> &v0, std::vector<value_d> &w0, int X, int Y, int Z, float visc, float dt, double &time_add_source, double &time_diffuse, double &time_project, double &time_advect, double &time_lin_solve, double &time_set_bnd) {
    auto time1 = std::chrono::high_resolution_clock::now();
    seq_add_source(u, u0, X, Y, Z, dt);
    seq_add_source(v, v0, X, Y, Z, dt);
    seq_add_source(w, w0, X, Y, Z, dt);
    auto time2 = std::chrono::high_resolution_clock::now();
    u0.swap(u);
    auto time3 = std::chrono::high_resolution_clock::now();
    seq_diffuse(u, u0, X, Y, Z, 1, visc, dt, time_lin_solve, time_set_bnd);
    auto time4 = std::chrono::high_resolution_clock::now();
    v0.swap(v);
    auto time5 = std::chrono::high_resolution_clock::now();
    seq_diffuse(v, v0, X, Y, Z, 2, visc, dt, time_lin_solve, time_set_bnd);
    auto time6 = std::chrono::high_resolution_clock::now();
    w0.swap(w);
    auto time7 = std::chrono::high_resolution_clock::now();
    seq_diffuse(w, w0, X, Y, Z, 3, visc, dt, time_lin_solve, time_set_bnd);
    auto time8 = std::chrono::high_resolution_clock::now();
    seq_project(u, v, w, u0, v0, X, Y, Z, time_lin_solve, time_set_bnd);
    auto time9 = std::chrono::high_resolution_clock::now();
    u0.swap(u);
    v0.swap(v);
    w0.swap(w);
    auto time10 = std::chrono::high_resolution_clock::now();
    seq_advect(u, u0, u0, v0, w0, X, Y, Z, 1, dt, time_set_bnd);
    seq_advect(v, v0, u0, v0, w0, X, Y, Z, 2, dt, time_set_bnd);
    seq_advect(w, w0, u0, v0, w0, X, Y, Z, 3, dt, time_set_bnd);
    auto time11 = std::chrono::high_resolution_clock::now();
    seq_project(u, v, w, u0, v0, X, Y, Z, time_lin_solve, time_set_bnd);
    auto time12 = std::chrono::high_resolution_clock::now();
    time_add_source += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time seq add source1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
    time_diffuse += std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();
    std::cout << "Time seq diffuse1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() << std::endl;
    time_diffuse += std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count();
    std::cout << "Time seq diffuse2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count() << std::endl;
    time_diffuse += std::chrono::duration_cast<std::chrono::microseconds>(time8 - time7).count();
    std::cout << "Time seq diffuse3:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time8 - time7).count() << std::endl;
    time_project += std::chrono::duration_cast<std::chrono::microseconds>(time9 - time8).count();
    std::cout << "Time seq project1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time9 - time8).count() << std::endl;
    time_advect += std::chrono::duration_cast<std::chrono::microseconds>(time11 - time10).count();
    std::cout << "Time seq advect1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time11 - time10).count() << std::endl;
    time_project += std::chrono::duration_cast<std::chrono::microseconds>(time12 - time11).count();
    std::cout << "Time seq project2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time12 - time11).count() << std::endl;
}

void seq_dens_step(std::vector<value_d> &x, std::vector<value_d> &x0, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, float diff, float dt, double &time_add_source, double &time_diffuse, double &time_advect, double &time_lin_solve, double &time_set_bnd) {
    auto time1 = std::chrono::high_resolution_clock::now();
    seq_add_source(x, x0, X, Y, Z, dt);
    auto time2 = std::chrono::high_resolution_clock::now();
    x0.swap(x);
    auto time3 = std::chrono::high_resolution_clock::now();
    seq_diffuse(x, x0, X, Y, Z, 0, diff, dt, time_lin_solve, time_set_bnd);
    auto time4 = std::chrono::high_resolution_clock::now();
    x0.swap(x);
    auto time5 = std::chrono::high_resolution_clock::now();
    seq_advect(x, x0, u, v, w, X, Y, Z, 0, dt, time_set_bnd);
    auto time6 = std::chrono::high_resolution_clock::now();
    time_add_source += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time seq add source2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
    time_diffuse += std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();
    std::cout << "Time seq diffuse4:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() << std::endl;
    time_advect += std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count();
    std::cout << "Time seq advect2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count() << std::endl;
}

void seq_add_source(std::vector<value_d> &x, std::vector<value_d> &s, int X, int Y, int Z, float dt) {
    int size = (X + 2) * (Y + 2) * (Z + 2);
    for (int i = 0; i < size; i++)
        x[i] += dt * s[i];
}

void seq_diffuse(std::vector<value_d> &x, std::vector<value_d> &x0, int X, int Y, int Z, int b, float diff, float dt, double &time_lin_solve, double &time_set_bnd) {
    int max = std::max(std::max(X, Y), std::max(Y, Z));
    float a = dt * diff * max * max * max;
    auto time1 = std::chrono::high_resolution_clock::now();
    seq_lin_solve(x, x0, X, Y, Z, b, a, 1 + 6 * a, time_set_bnd);
    auto time2 = std::chrono::high_resolution_clock::now();
    time_lin_solve += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time seq lin solve1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
}

void seq_advect(std::vector<value_d> &d, std::vector<value_d> &d0, std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, int X, int Y, int Z, int b, float dt, double &time_set_bnd) {
    int max = std::max(std::max(X, Y), std::max(Y, Z));
    value_d dtx = dt * max;
    value_d dty = dt * max;
    value_d dtz = dt * max;

    for (int i = 1; i <= X; i++) {
        for (int j = 1; j <= Y; j++) {
            for (int k = 1; k <= Z; k++) {
                value_d x = i - dtx * u[IX(i, j, k)];
                value_d y = j - dty * v[IX(i, j, k)];
                value_d z = k - dtz * w[IX(i, j, k)];
                if (x < 0.5f)
                    x = 0.5f;
                if (x > X + 0.5f)
                    x = X + 0.5f;
                int i0 = (int)x;
                int i1 = i0 + 1;
                if (y < 0.5f)
                    y = 0.5f;
                if (y > Y + 0.5f)
                    y = Y + 0.5f;
                int j0 = (int)y;
                int j1 = j0 + 1;
                if (z < 0.5f)
                    z = 0.5f;
                if (z > Z + 0.5f)
                    z = Z + 0.5f;
                int k0 = (int)z;
                int k1 = k0 + 1;

                value_d s1 = x - i0;
                value_d s0 = 1 - s1;
                value_d t1 = y - j0;
                value_d t0 = 1 - t1;
                value_d u1 = z - k0;
                value_d u0 = 1 - u1;
                d[IX(i, j, k)] = s0 * (t0 * u0 * d0[IX(i0, j0, k0)] + t1 * u0 * d0[IX(i0, j1, k0)] + t0 * u1 * d0[IX(i0, j0, k1)] + t1 * u1 * d0[IX(i0, j1, k1)]) +
                                 s1 * (t0 * u0 * d0[IX(i1, j0, k0)] + t1 * u0 * d0[IX(i1, j1, k0)] + t0 * u1 * d0[IX(i1, j0, k1)] + t1 * u1 * d0[IX(i1, j1, k1)]);
            }
        }
    }
    auto time1 = std::chrono::high_resolution_clock::now();
    seq_set_bnd(d, X, Y, Z, b);
    auto time2 = std::chrono::high_resolution_clock::now();
    time_set_bnd += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time seq set bnd1:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
}

void seq_project(std::vector<value_d> &u, std::vector<value_d> &v, std::vector<value_d> &w, std::vector<value_d> &p, std::vector<value_d> &div, int X, int Y, int Z, double &time_lin_solve, double &time_set_bnd) {
    for (int i = 1; i <= X; i++) {
        for (int j = 1; j <= Y; j++) {
            for (int k = 1; k <= Z; k++) {
                div[IX(i, j, k)] = -1.0 / 3.0 * ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) / X + (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) / X + (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) / X);
                p[IX(i, j, k)] = 0;
            }
        }
    }
    auto time1 = std::chrono::high_resolution_clock::now();
    seq_set_bnd(div, X, Y, Z, 0);
    seq_set_bnd(p, X, Y, Z, 0);
    auto time2 = std::chrono::high_resolution_clock::now();

    seq_lin_solve(p, div, X, Y, Z, 0, 1, 6, time_set_bnd);
    auto time3 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i <= X; i++) {
        for (int j = 1; j <= Y; j++) {
            for (int k = 1; k <= Z; k++) {
                u[IX(i, j, k)] -= 0.5f * X * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[IX(i, j, k)] -= 0.5f * X * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[IX(i, j, k)] -= 0.5f * X * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }
    auto time4 = std::chrono::high_resolution_clock::now();
    seq_set_bnd(u, X, Y, Z, 1);
    seq_set_bnd(v, X, Y, Z, 2);
    seq_set_bnd(w, X, Y, Z, 3);
    auto time5 = std::chrono::high_resolution_clock::now();
    time_set_bnd += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time seq set bnd2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
    time_lin_solve += std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
    std::cout << "Time seq lin solve2:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() << std::endl;
    time_set_bnd += std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count();
    std::cout << "Time seq set bnd3:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() << std::endl;
}

void seq_lin_solve(std::vector<value_d> &x, std::vector<value_d> &x0, int X, int Y, int Z, int b, float a, float c, double &time_set_bnd) {
    // iterate the solver
    for (int l = 0; l < LINEARSOLVERTIMES; l++) {
        // update for each cell
        for (int i = 1; i <= X; i++) {
            for (int j = 1; j <= Y; j++) {
                for (int k = 1; k <= Z; k++) {
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
                }
            }
        }
        auto time1 = std::chrono::high_resolution_clock::now();
        seq_set_bnd(x, X, Y, Z, b);
        auto time2 = std::chrono::high_resolution_clock::now();
        time_set_bnd += std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
        std::cout << "Time seq set bnd4:\t" << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
    }
}

void seq_set_bnd(std::vector<value_d> &x, int X, int Y, int Z, int b) {
    //setting faces
    for (int i = 1; i <= X; i++) {
        for (int j = 1; j <= Y; j++) {
            x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, Z + 1)] = b == 3 ? -x[IX(i, j, Z)] : x[IX(i, j, Z)];
        }
    }

    for (int i = 1; i <= Y; i++) {
        for (int j = 1; j <= Z; j++) {
            x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
            x[IX(X + 1, i, j)] = b == 1 ? -x[IX(X, i, j)] : x[IX(X, i, j)];
        }
    }

    for (int i = 1; i <= X; i++) {
        for (int j = 1; j <= Z; j++) {
            x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
            x[IX(i, Y + 1, j)] = b == 2 ? -x[IX(i, Y, j)] : x[IX(i, Y, j)];
        }
    }

    //Setting edges
    for (int i = 1; i <= X; i++) {
        x[IX(i, 0, 0)] = 1.0 / 2.0 * (x[IX(i, 1, 0)] + x[IX(i, 0, 1)]);
        x[IX(i, Y + 1, 0)] = 1.0 / 2.0 * (x[IX(i, Y, 0)] + x[IX(i, Y + 1, 1)]);
        x[IX(i, 0, Z + 1)] = 1.0 / 2.0 * (x[IX(i, 0, Z)] + x[IX(i, 1, Z + 1)]);
        x[IX(i, Y + 1, Z + 1)] = 1.0 / 2.0 * (x[IX(i, Y, Z + 1)] + x[IX(i, Y + 1, Z)]);
    }

    for (int i = 1; i <= Y; i++) {
        x[IX(0, i, 0)] = 1.0 / 2.0 * (x[IX(1, i, 0)] + x[IX(0, i, 1)]);
        x[IX(X + 1, i, 0)] = 1.0 / 2.0 * (x[IX(X, i, 0)] + x[IX(X + 1, i, 1)]);
        x[IX(0, i, Z + 1)] = 1.0 / 2.0 * (x[IX(0, i, Z)] + x[IX(1, i, Z + 1)]);
        x[IX(X + 1, i, Z + 1)] = 1.0 / 2.0 * (x[IX(X, i, Z + 1)] + x[IX(X + 1, i, Z)]);
    }

    for (int i = 1; i <= Z; i++) {
        x[IX(0, 0, i)] = 1.0 / 2.0 * (x[IX(0, 1, i)] + x[IX(1, 0, i)]);
        x[IX(0, Y + 1, i)] = 1.0 / 2.0 * (x[IX(0, Y, i)] + x[IX(1, Y + 1, i)]);
        x[IX(X + 1, 0, i)] = 1.0 / 2.0 * (x[IX(X, 0, i)] + x[IX(X + 1, 1, i)]);
        x[IX(X + 1, Y + 1, i)] = 1.0 / 2.0 * (x[IX(X + 1, Y, i)] + x[IX(X, Y + 1, i)]);
    }

    //setting corners
    x[IX(0, 0, 0)] = 1.0 / 3.0 * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(0, Y + 1, 0)] = 1.0 / 3.0 * (x[IX(1, Y + 1, 0)] + x[IX(0, Y, 0)] + x[IX(0, Y + 1, 1)]);

    x[IX(X + 1, 0, 0)] = 1.0 / 3.0 * (x[IX(X, 0, 0)] + x[IX(X + 1, 1, 0)] + x[IX(X + 1, 0, 1)]);
    x[IX(X + 1, Y + 1, 0)] = 1.0 / 3.0 * (x[IX(X, Y + 1, 0)] + x[IX(X + 1, Y, 0)] + x[IX(X + 1, Y + 1, 1)]);

    x[IX(0, 0, Z + 1)] = 1.0 / 3.0 * (x[IX(1, 0, Z + 1)] + x[IX(0, 1, Z + 1)] + x[IX(0, 0, Z)]);
    x[IX(0, Y + 1, Z + 1)] = 1.0 / 3.0 * (x[IX(1, Y + 1, Z + 1)] + x[IX(0, Y, Z + 1)] + x[IX(0, Y + 1, Z)]);

    x[IX(X + 1, 0, Z + 1)] = 1.0 / 3.0 * (x[IX(X, 0, Z + 1)] + x[IX(X + 1, 1, Z + 1)] + x[IX(X + 1, 0, Z)]);
    x[IX(X + 1, Y + 1, Z + 1)] = 1.0 / 3.0 * (x[IX(X, Y + 1, Z + 1)] + x[IX(X + 1, Y, Z + 1)] + x[IX(X + 1, Y + 1, Z)]);
}

// helper functions
void print_array(const std::vector<value_d> &array, int X, int Y, int Z) {
    for (int i = 0; i < X + 2; i++) {
        for (int j = 0; j < Y + 2; j++) {
            for (int k = 0; k < Z + 2; k++) {
                std::cout << std::fixed << std::setprecision(3) << array[IX(i, j, k)] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

bool check_parameters(int argc, char *argv[]) {
    if (argc < 5) {
        return false;
    }

    bool is_numeric = true;
    for (size_t i = 1; i < 5; i++) {
        for (size_t j = 0; j < strlen(argv[i]); j++)
            if (!isdigit(argv[i][j]))
                is_numeric = false;
    }

    if (argc >= 6) {
        for (size_t i = 0; i < argc; i++) {
            if (!strcmp(argv[i], "-s")) {
                sequential_check = true;
            } else if (!strcmp(argv[i], "-pds")) {
                print_seq_result_dens = true;
            } else if (!strcmp(argv[i], "-pd")) {
                print_result_dens = true;
            } else if (!strcmp(argv[i], "-h")) {
                print_help();
            }
        }
    }

    return is_numeric;
}

void print_help() {
    std::cout << "      OPTIONAL PARAMETERS:" << std::endl;
    std::cout << std::endl;
    std::cout << "-s    enable sequential check (including time measurment)" << std::endl;
    std::cout << "-pds  print dens-buffer sequential result" << std::endl;
    std::cout << "-pd   print dens-buffer gpu result" << std::endl;
    std::cout << "-h    show this help menu" << std::endl;
}
