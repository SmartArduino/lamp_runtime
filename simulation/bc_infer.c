#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bc_model.h"

static void fc(const float* x, const float* W, const float* b,
               int in, int out, float* y, int relu) {
    for (int j = 0; j < out; j++) {
        const float* w = W + (size_t)j * in;
        float s = b[j];
        for (int i = 0; i < in; i++) s += x[i] * w[i];
        y[j] = (relu && s < 0.0f) ? 0.0f : s;
    }
}

static void mat_vec3(const float R[9], const float v[3], float out[3]) {
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

static void rot_z(float t, float R[9]) {
    float c = cosf(t), s = sinf(t);
    R[0] = c; R[1] = -s; R[2] = 0;
    R[3] = s; R[4] = c; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
}

static void rot_x(float t, float R[9]) {
    float c = cosf(t), s = sinf(t);
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = c; R[5] = -s;
    R[6] = 0; R[7] = s; R[8] = c;
}

static void mul33(const float A[9], const float B[9], float C[9]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}

void head_fk(const float q[5], float out[3]) {
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float Rt[9], tmp[3];
    float p[3] = {0, 0, 0.01f};

    float p1[3] = {0, 0, 0.041f};
    mat_vec3(R, p1, tmp); p[0] += tmp[0]; p[1] += tmp[1]; p[2] += tmp[2];
    rot_z(q[0], Rt); mul33(R, Rt, R);

    float p2[3] = {0.002f, 0, 0.04f};
    mat_vec3(R, p2, tmp); p[0] += tmp[0]; p[1] += tmp[1]; p[2] += tmp[2];
    rot_x(q[1], Rt); mul33(R, Rt, R);

    float p3[3] = {0.005f, -0.005f, 0.177f};
    mat_vec3(R, p3, tmp); p[0] += tmp[0]; p[1] += tmp[1]; p[2] += tmp[2];
    rot_x(q[2], Rt); mul33(R, Rt, R);

    float p4[3] = {0, 0, 0.139f};
    mat_vec3(R, p4, tmp); p[0] += tmp[0]; p[1] += tmp[1]; p[2] += tmp[2];
    rot_z(q[3], Rt); mul33(R, Rt, R);

    float p5[3] = {0, 0, 0.033f};
    mat_vec3(R, p5, tmp); p[0] += tmp[0]; p[1] += tmp[1]; p[2] += tmp[2];

    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
}

void bc_predict(const float qpos[5], const float qvel[5], const float cup[3],
                float action[5]) {
    float head[3];
    head_fk(qpos, head);
    float state[BC_IN_DIM];
    for (int i = 0; i < 5; i++) state[i] = qpos[i];
    for (int i = 0; i < 5; i++) state[5 + i] = qvel[i];
    for (int i = 0; i < 3; i++) state[10 + i] = head[i];
    for (int i = 0; i < 3; i++) state[13 + i] = cup[i];

    float h1[BC_H1], h2[BC_H2], delta[5];
    fc(state, W1, b1, BC_IN_DIM, BC_H1, h1, 1);
    fc(h1, W2, b2, BC_H1, BC_H2, h2, 1);
    fc(h2, W3, b3, BC_H2, BC_OUT, delta, 0);
    for (int j = 0; j < 5; j++) {
        float a = qpos[j] + delta[j];
        if (a < -1.57f) a = -1.57f;
        if (a > 1.57f) a = 1.57f;
        action[j] = a;
    }
}

#ifdef STANDALONE
int main(int argc, char** argv) {
    float qpos[5] = {0}, qvel[5] = {0}, cup[3] = {0.0f, -0.23f, 0.30f};
    if (argc > 13) {
        for (int i = 0; i < 5; i++) qpos[i] = (float)atof(argv[i + 1]);
        for (int i = 0; i < 5; i++) qvel[i] = (float)atof(argv[i + 6]);
        for (int i = 0; i < 3; i++) cup[i] = (float)atof(argv[i + 11]);
    }
    float head[3], action[5];
    head_fk(qpos, head);
    printf("head_fk: %.4f %.4f %.4f\n", head[0], head[1], head[2]);
    bc_predict(qpos, qvel, cup, action);
    printf("action : ");
    for (int j = 0; j < 5; j++) printf("%.4f ", action[j]);
    printf("\n");
    return 0;
}
#endif
