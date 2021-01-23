#include <celerity/celerity.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

typedef float value_d;

#define LINEARSOLVERTIMES 10
#define PORT 9999
#define IX(i, j, k) ((X + 2) * (Y + 2) * (i) + (X + 2) * (j) + (k))

// celerity functions
void get_force_source(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z);

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

void send_data(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, int socket, int X, int Y, int Z);

void receive_data(celerity::distr_queue &queue, int socket, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z);

// helper functions
bool check_parameters(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    celerity::runtime::init(&argc, &argv);

    if (!check_parameters(argc, argv)) {
        std::cout << "expected input: ./[file_name] timesteps height width depth" << std::endl;
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

    // ---------- server setup ----------
    int server_fd;
    int new_socket;
    int opt = 1;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    std::cout << "Waiting for Client ..." << std::endl;

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    std::cout << "Connection established" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    {
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

        bool add_source = true;
        bool add_force_x = false;
        bool add_force_y = true;
        bool add_force_z = false;

        for (int i = 0; i < timesteps; i++) {
            get_force_source(queue, dens_prev_buf, u_prev_buf, v_prev_buf, w_prev_buf, X, Y, Z, force, source, add_source, add_force_x, add_force_y, add_force_z);
            velocity_step(queue, u_buf, v_buf, w_buf, u_prev_buf, v_prev_buf, w_prev_buf, X, Y, Z, visc, dt);
            density_step(queue, dens_buf, dens_prev_buf, u_buf, v_buf, w_buf, X, Y, Z, diff, dt);
            send_data(queue, dens_buf, new_socket, X, Y, Z);
            receive_data(queue, new_socket, X, Y, Z, force, source, add_source, add_force_x, add_force_y, add_force_z);
        }

        queue.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {
            cgh.host_task(celerity::on_master_node, [=]() {
                celerity::experimental::bench::end("main program");
                std::cout << "Simulation finished" << std::endl;
            });
        });
    }

    return EXIT_SUCCESS;
}

// celerity functions
void get_force_source(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z) {
    queue.submit(celerity::allow_by_ref, [&, X, Y, Z, force, source](celerity::handler &cgh) {
        auto d_prev = d.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        auto u_prev = u.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        auto v_prev = v.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        auto w_prev = w.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        cgh.host_task(celerity::on_master_node, [=, &add_source, &add_force_x, &add_force_y, &add_force_z]() {
            for (size_t i = 0; i < X + 2; i++) {
                for (size_t j = 0; j < Y + 2; j++) {
                    for (size_t k = 0; k < Z + 2; k++) {
                        d_prev[{i, j, k}] = 0;
                        u_prev[{i, j, k}] = 0;
                        v_prev[{i, j, k}] = 0;
                        w_prev[{i, j, k}] = 0;
                    }
                }
            }

            if (add_force_x) {
                size_t i = 2;
                size_t j = Y / 2;
                size_t k = Z / 2;

                if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z) return;
                u_prev[{i, j, k}] = force * 10;
                add_force_x = false;
            }

            if (add_force_y) {
                size_t i = X / 2;
                size_t j = 2;
                size_t k = Z / 2;

                if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z) return;
                v_prev[{i, j, k}] = force * 10;
                add_force_y = false;
            }

            if (add_force_z) {
                size_t i = X / 2;
                size_t j = Y / 2;
                size_t k = 2;

                if (i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z) return;
                w_prev[{i, j, k}] = force * 10;
                add_force_z = false;
            }

            if (add_source) {
                size_t i = X / 2;
                size_t j = Y / 2;
                size_t k = Z / 2;
                d_prev[{i, j, k}] = source;
                add_source = false;
            }
        });
    });
}

void velocity_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, celerity::buffer<value_d, 3> &u0, celerity::buffer<value_d, 3> &v0, celerity::buffer<value_d, 3> &w0, int X, int Y, int Z, value_d visc, value_d dt) {
    add_source(queue, u, u0, X, Y, Z, dt);
    add_source(queue, v, v0, X, Y, Z, dt);
    add_source(queue, w, w0, X, Y, Z, dt);
    queue.slow_full_sync();
    std::swap(u0, u);
    diffuse(queue, u, u0, X, Y, Z, 1, visc, dt);
    queue.slow_full_sync();
    std::swap(v0, v);
    diffuse(queue, v, v0, X, Y, Z, 2, visc, dt);
    queue.slow_full_sync();
    std::swap(w0, w);
    diffuse(queue, w, w0, X, Y, Z, 3, visc, dt);
    project(queue, u, v, w, u0, v0, X, Y, Z);
    queue.slow_full_sync();
    std::swap(u0, u);
    std::swap(v0, v);
    std::swap(w0, w);
    advect(queue, u, u0, u0, v0, w0, X, Y, Z, 1, dt);
    advect(queue, v, v0, u0, v0, w0, X, Y, Z, 2, dt);
    advect(queue, w, w0, u0, v0, w0, X, Y, Z, 3, dt);
    project(queue, u, v, w, u0, v0, X, Y, Z);
}

void density_step(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, celerity::buffer<value_d, 3> &u, celerity::buffer<value_d, 3> &v, celerity::buffer<value_d, 3> &w, int X, int Y, int Z, int diff, value_d dt) {
    add_source(queue, x, x0, X, Y, Z, dt);
    queue.slow_full_sync();
    std::swap(x0, x);
    diffuse(queue, x, x0, X, Y, Z, 0, diff, dt);
    queue.slow_full_sync();
    std::swap(x0, x);
    advect(queue, x, x0, u, v, w, X, Y, Z, 0, dt);
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
    lin_solve(queue, x, x0, X, Y, Z, b, a, 1 + 6 * a);
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

    set_bnd(queue, x, X, Y, Z, b);
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

    set_bnd(queue, v0, X, Y, Z, 0);
    set_bnd(queue, u0, X, Y, Z, 0);

    lin_solve(queue, u0, v0, X, Y, Z, 0, 1, 6);

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

    set_bnd(queue, u, X, Y, Z, 1);
    set_bnd(queue, v, X, Y, Z, 2);
    set_bnd(queue, w, X, Y, Z, 3);
}

// Host task version
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
        set_bnd(queue, x, X, Y, Z, b);
    }
}

/*
// Parallel verion 
void lin_solve(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, celerity::buffer<value_d, 3> &x0, int X, int Y, int Z, int b, value_d a, value_d c) {
    for (size_t i = 0; i < LINEARSOLVERTIMES; i++) {
        queue.submit([=](celerity::handler &cgh) {
            auto dens = x.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::neighborhood<3>(1, 1, 1));
            auto dens_prev = x0.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<3>());
            cgh.parallel_for<class linSolve>(cl::sycl::range<3>(X, Y, Z), cl::sycl::id<3>(1, 1, 1), [=](cl::sycl::item<3> item) {
                dens[item] = (dens_prev[item] + a * (dens[{item[0] - 1, item[1], item[2]}] + dens[{item[0] + 1, item[1], item[2]}] + dens[{item[0], item[1] - 1, item[2]}] + dens[{item[0], item[1] + 1, item[2]}] + dens[{item[0], item[1], item[2] - 1}] + dens[{item[0], item[1], item[2] + 1}])) / c;
            });
        });
        set_bnd(queue, x, X, Y, Z, b);
    }
}
*/

void set_bnd(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &x, int X, int Y, int Z, int b) {
    /*
    // Parallel verion 
    set_bnd_faces(queue, x, X, Y, Z, b);
    set_bnd_edges(queue, x, X, Y, Z, b);
    set_bnd_corners(queue, x, X, Y, Z, b);
    */

    // Host task version
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
    });
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

void send_data(celerity::distr_queue &queue, celerity::buffer<value_d, 3> &d, int socket, int X, int Y, int Z) {
    queue.submit(celerity::allow_by_ref, [&, X, Y, Z, socket](celerity::handler &cgh) {
        auto dens = d.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<3>());
        cgh.host_task(celerity::on_master_node, [=]() {
            std::vector<value_d> result((X + 2) * (Y + 2) * (Z + 2));
            for (size_t i = 0; i < X + 2; i++) {
                for (size_t j = 0; j < Y + 2; j++) {
                    for (size_t k = 0; k < Z + 2; k++) {
                        result[IX(i, j, k)] = dens[{i, j, k}];
                    }
                }
            }
            //sending
            int bytes = send(socket, (char *)result.data(), sizeof(value_d) * result.size(), 0);
            // std::cout << "Bytes sent: " << bytes << std::endl;
        });
    });
}

void receive_data(celerity::distr_queue &queue, int socket, int X, int Y, int Z, float force, float source, bool &add_source, bool &add_force_x, bool &add_force_y, bool &add_force_z) {
    queue.submit(celerity::allow_by_ref, [&, X, Y, Z, force, source, socket](celerity::handler &cgh) {
        cgh.host_task(celerity::on_master_node, [=, &add_source, &add_force_x, &add_force_y, &add_force_z]() {
            char msg[1];
            int bytes = recv(socket, msg, 1, MSG_DONTWAIT);

            if (bytes > 0) {
                std::cout << "Received signal: " << msg[0] << std::endl;
                switch (msg[0]) {
                    case 'x':
                        add_force_x = true;
                        break;
                    case 'y':
                        add_force_y = true;
                        break;
                    case 'z':
                        add_force_z = true;
                        break;
                    case 's':
                        add_source = true;
                        break;
                }
            } else if (bytes == -1) {
                if (!(errno == EAGAIN || errno == EWOULDBLOCK)) {
                    std::cout << "Error while receiving data" << std::endl;
                    fprintf(stderr, "recv: %s (%d)\n", strerror(errno), errno);
                }
            }
        });
    });
}

// helper functions
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

    return is_numeric;
}
