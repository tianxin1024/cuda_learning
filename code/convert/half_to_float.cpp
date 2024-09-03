// g++ -o half_to_float half_to_float.cpp
#include <iostream>

static float float16_to_float32(unsigned short value) {
    // [float16] 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;     // 符号位
    unsigned short exponent = (value & 0x7c00) >> 10; // 指数位
    unsigned short significand = value & 0x03FF;      // 尾数位

    // [float32] 1 : 8 : 23
    union {
        unsigned int u;
        float f;
    } tmp;

    if (exponent == 0) {
        if (significand == 0) {
            // zero
            tmp.u = (sign << 31);
        } else {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0) {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    } else if (exponent == 0x1F) {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    } else {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

int main() {
    // test
    unsigned short arr = 234;
    std::cout << float16_to_float32(arr) << std::endl;
    std::cout << float16_to_float32(0x7FFF) << std::endl;
    std::cout << float16_to_float32(0x7C01) << std::endl;
    std::cout << float16_to_float32(0x7FFF) << std::endl;
    return 0;
}
