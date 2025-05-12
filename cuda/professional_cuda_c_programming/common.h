#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const mcError_t error = call;                                            \
    if (error != mcSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                mcGetErrorString(error));                                    \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline double nanoseconds() 
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // 获取系统实时时间
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

#endif // _COMMON_H