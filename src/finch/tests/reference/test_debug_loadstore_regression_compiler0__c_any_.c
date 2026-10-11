
#ifdef _WIN32
    #define FINCH_EXPORT __declspec( dllexport )
#else
    #define FINCH_EXPORT
#endif

#include <stdint.h>
typedef void* (*fptr)( void**, uint64_t );
struct CNumpyBuffer {
    void* arr;
    void* data;
    uint64_t length;
    fptr resize;
};
FINCH_EXPORT int64_t finch_access(struct CNumpyBuffer*, int64_t);
FINCH_EXPORT int64_t finch_change(struct CNumpyBuffer*, int64_t, int64_t);
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
FINCH_EXPORT int64_t finch_access(struct CNumpyBuffer* a, int64_t idx) {
    struct CNumpyBuffer* a_ = a;
    int64_t* a__data = (int64_t*)a_->data;
    size_t a__length = a_->length;
    if (!(idx >= (int64_t)0 & idx < a__length)) {
        fputs("Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))\n", stderr);
        exit(1);
    }
    int64_t val = (a__data)[idx];
    if (!(idx >= (int64_t)0 & idx < a__length)) {
        fputs("Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))\n", stderr);
        exit(1);
    }
    int64_t val2 = (a__data)[idx];
    return val;
}

FINCH_EXPORT int64_t finch_change(struct CNumpyBuffer* a, int64_t idx, int64_t val) {
    struct CNumpyBuffer* a_ = a;
    int64_t* a__data_2 = (int64_t*)a_->data;
    size_t a__length_2 = a_->length;
    if (!(idx >= (int64_t)0 & idx < a__length_2)) {
        fputs("Finch assertion failed: and_(ge(idx, 0), lt(idx, length(slot(a_, np_buf_t(int64)))))\n", stderr);
        exit(1);
    }
    (a__data_2)[idx] = val;
    return (int64_t)0;
}
