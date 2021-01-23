#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Server-Client-includes
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Mswsock.lib")
#pragma comment(lib, "AdvApi32.lib")

#define PORT "9999"
SOCKET win_socket;
const int size = 20;
const int total_data = (size + 2) * (size + 2) * (size + 2) * sizeof(float);
float* density;
char* msg_buf;
bool simulation = true;

enum State {
    X_FORCE,
    Y_FORCE,
    Z_FORCE,
    SOURCE
};

#define IX(i, j, k) ((i) + (size + 2) * (j) + (size + 2) * (size + 2) * (k))

float source_alpha = 0.05f;
static int omx, omy, mx, my;
bool rot_on = false;
static int mouse_down[3];
float rot0 = 30;
float rot1 = -45;

// Network related functions
void network_receive_data(SOCKET win_socket, char* buf) {
    int bytes = 0;
    int total_bytes = 0;
    do {
        bytes = recv(win_socket, buf + total_bytes, total_data - total_bytes, 0);
        if (bytes > 0) {
            total_bytes += bytes;
        }
        else if (bytes == -1)
            std::cout << "Error while receiving data. Error Code: " << WSAGetLastError() << std::endl;
        else if (bytes == 0) {
            std::cout << "No Bytes" << std::endl;
            simulation = false;
        }
    } while (total_bytes < total_data && simulation);
    // std::cout << "Bytes received: " << total_bytes << std::endl;
}

void network_send_data(SOCKET win_socket, State s) {
    char state_msg[1];
    switch (s) {
    case X_FORCE:
        state_msg[0] = 'x';
        break;
    case Y_FORCE:
        state_msg[0] = 'y';
        break;
    case Z_FORCE:
        state_msg[0] = 'z';
        break;
    case SOURCE:
        state_msg[0] = 's';
        break;
    default:
        break;
    }
    int bytes = send(win_socket, state_msg, 1, 0);

    if (bytes > 0)
        std::cout << "Bytes sent: " << bytes << std::endl;
    else if (bytes < 0)
        std::cout << "Error while sending data. Error Code: " << WSAGetLastError() << std::endl;
}

int network_setup(SOCKET& win_socket) {
    WSADATA wsaData;
    win_socket = INVALID_SOCKET;
    struct addrinfo* result = NULL,
        * ptr = NULL,
        hints;
    int ret_val;

    // Initialize Winsock
    ret_val = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (ret_val != 0) {
        printf("WSAStartup failed with error: %d\n", ret_val);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    ret_val = getaddrinfo("localhost", PORT, &hints, &result);
    if (ret_val != 0) {
        printf("getaddrinfo failed with error: %d\n", ret_val);
        WSACleanup();
        return 1;
    }

    ptr = result;
    // Create a SOCKET for connecting to server
    win_socket = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
    if (win_socket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }

    // Connect to server.
    ret_val = connect(win_socket, ptr->ai_addr, (int)ptr->ai_addrlen);
    if (ret_val == SOCKET_ERROR) {
        closesocket(win_socket);
        win_socket = INVALID_SOCKET;
    }

    freeaddrinfo(result);

    if (win_socket == INVALID_SOCKET) {
        printf("Unable to connect to server!\n");
        WSACleanup();
        getchar();
        return 1;
    }
}

// Simulation related functions
void update() {
    float dx = omx - mx;
    float dy = omy - my;

    rot1 += dx;
}

void draw_density(float* dens) {
    int M = size - 1;
    int N = size - 1;
    int O = size - 1;

    float x, y, z, h, d000, d010, d100, d110, d001, d011, d101, d111;

    h = 1.0f / std::max(std::max(M, N), std::max(N, O));

    glBegin(GL_QUADS);

    for (int i = 0; i <= M; i++) {
        x = (i - 0.5f) * h;
        for (int j = 0; j <= N; j++) {
            y = (j - 0.5f) * h;
            for (int k = 0; k <= O; k++) {
                z = (k - 0.5f) * h;

                d000 = dens[IX(i, j, k)];
                d010 = dens[IX(i, j + 1, k)];
                d100 = dens[IX(i + 1, j, k)];
                d110 = dens[IX(i + 1, j + 1, k)];

                d001 = dens[IX(i, j, k + 1)];
                d011 = dens[IX(i, j + 1, k + 1)];
                d101 = dens[IX(i + 1, j, k + 1)];
                d111 = dens[IX(i + 1, j + 1, k + 1)];

                // draw density as a cube of quads
                glColor4f(d111, d111, d111, source_alpha);
                glVertex3f(x + h, y + h, z + h);
                glColor4f(d011, d011, d011, source_alpha);
                glVertex3f(x, y + h, z + h);
                glColor4f(d001, d001, d001, source_alpha);
                glVertex3f(x, y, z + h);
                glColor4f(d101, d101, d101, source_alpha);
                glVertex3f(x + h, y, z + h);

                glColor4f(d110, d110, d110, source_alpha);
                glVertex3f(x + h, y + h, z);
                glColor4f(d111, d111, d111, source_alpha);
                glVertex3f(x + h, y + h, z + h);
                glColor4f(d101, d101, d101, source_alpha);
                glVertex3f(x + h, y, z + h);
                glColor4f(d100, d100, d100, source_alpha);
                glVertex3f(x + h, y, z);

                glColor4f(d010, d010, d010, source_alpha);
                glVertex3f(x, y + h, z);
                glColor4f(d110, d110, d110, source_alpha);
                glVertex3f(x + h, y + h, z);
                glColor4f(d100, d100, d100, source_alpha);
                glVertex3f(x + h, y, z);
                glColor4f(d000, d000, d000, source_alpha);
                glVertex3f(x, y, z);

                glColor4f(d011, d011, d011, source_alpha);
                glVertex3f(x, y + h, z + h);
                glColor4f(d010, d010, d010, source_alpha);
                glVertex3f(x, y + h, z);
                glColor4f(d000, d000, d000, source_alpha);
                glVertex3f(x, y, z);
                glColor4f(d001, d001, d001, source_alpha);
                glVertex3f(x, y, z + h);

                glColor4f(d100, d100, d100, source_alpha);
                glVertex3f(x + h, y, z);
                glColor4f(d000, d000, d000, source_alpha);
                glVertex3f(x, y, z);
                glColor4f(d001, d001, d001, source_alpha);
                glVertex3f(x, y, z + h);
                glColor4f(d101, d101, d101, source_alpha);
                glVertex3f(x + h, y, z + h);

                glColor4f(d110, d110, d110, source_alpha);
                glVertex3f(x + h, y + h, z);
                glColor4f(d010, d010, d010, source_alpha);
                glVertex3f(x, y + h, z);
                glColor4f(d011, d011, d011, source_alpha);
                glVertex3f(x, y + h, z + h);
                glColor4f(d111, d111, d111, source_alpha);
                glVertex3f(x + h, y + h, z + h);
            }
        }
    }

    glEnd();
}

static void draw_axis() {
    glLineWidth(1.0f);
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);

    glEnd();
}

void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    glRotatef(rot0, 1.0f, 0.0f, 0.0f);
    glRotatef(rot1, 0.0f, 1.0f, 0.0f);

    draw_density(density);
    draw_axis();

    glEnd();
    glPopMatrix();
    glFlush();
    glutSwapBuffers();
}

int init(void) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    network_receive_data(win_socket, msg_buf);
    density = (float*)msg_buf;
    return (1);
}

static void key_func(unsigned char key, int x, int y) {
    switch (key) {
    case 27:  // ESC key
        glutLeaveMainLoop();
        break;
    case 'c':
    case 'C':
        if (rot_on)
            rot_on = false;
        else
            rot_on = true;
        break;
    case 'x':
    case 'X':
        network_send_data(win_socket, X_FORCE);
        break;
    case 'y':
    case 'Y':
        network_send_data(win_socket, Y_FORCE);
        break;
    case 'z':
    case 'Z':
        network_send_data(win_socket, Z_FORCE);
        break;
    case 's':
    case 'S':
        network_send_data(win_socket, SOURCE);
        break;
    }
}

static void motion_func(int x, int y) {
    omx = mx;
    omy = my;
    mx = x;
    my = y;
    float dx = (omx - mx) * 0.1;
    float dy = (omy - my) * 0.1;
    rot1 -= dx;
    rot0 -= dy;
}

static void mouse_func(int button, int state, int x, int y) {
    omx = mx = x;
    omx = my = y;

    mouse_down[button] = state == GLUT_DOWN;
}

static void idle_func(void) {
    if (rot_on) {
        update();
    }
    if (simulation) {
        network_receive_data(win_socket, msg_buf);
        density = (float*)msg_buf;
    }

    glutPostRedisplay();
}

static void open_glut_window(void) {
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutInitWindowSize(1200, 800);
    glutInitWindowPosition(400, 200);
    glutCreateWindow("3D_FLUID");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();

    glutKeyboardFunc(key_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);

    glutIdleFunc(idle_func);
    glutDisplayFunc(display);
}



int main(int argc, char** argv) {
    density = (float*)calloc((size + 2) * (size + 2) * (size + 2), sizeof(float));
    msg_buf = (char*)calloc((size + 2) * (size + 2) * (size + 2) * sizeof(float), sizeof(char));
    density = (float*)msg_buf;
    network_setup(win_socket);

    glutInit(&argc, argv);
    open_glut_window();
    init();
    glutMainLoop();

    return EXIT_SUCCESS;
}