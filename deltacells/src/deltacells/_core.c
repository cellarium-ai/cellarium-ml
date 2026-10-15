// Copyright Contributors to the Cellarium project.
// SPDX-License-Identifier: BSD-3-Clause

/*
 * deltacells C core.
 *
 * Operations that release the GIL while they run so that Python threads can run them in parallel:
 *
 *   decode_chunk    zstd-decompress one chunk (a gap stream and a value stream), un-interleave the byte planes and
 *                   prefix-sum the per-cell gene gaps into gene indices.
 *   apply_exceptions patch the entries of a decoded chunk whose value or gap did not fit in a uint16 (rare; only called for
 *                   chunks that have an exception frame).
 *   gather_rows     copy selected rows of a CSR matrix into caller-provided buffers (optionally converting the
 *                   values from uint32 to float32 on the way).
 *   compress_stream interleave -> byte-plane a uint16 array and zstd-compress it.
 *   compress_bytes  plain zstd compression (the exception frames).
 *
 * Buffers are taken through the buffer protocol (numpy arrays, bytes, memoryviews, mmaps ...); nothing here depends
 * on numpy or torch headers. All integers are little-endian and native-aligned; misaligned buffers are rejected.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#define ALIGNED(p, a) (((uintptr_t)(p) % (a)) == 0)

/* ------------------------------------------------------------------------------------------------ decode_chunk */

enum {
    DC_OK = 0,
    DC_ZSTD_ERROR = 1,
    DC_SIZE_MISMATCH = 2,
    DC_BAD_INDPTR = 3,
    DC_INDEX_OUT_OF_RANGE = 4,
    DC_NO_MEMORY = 5
};

static int decode_chunk_impl(const uint8_t *blob, uint64_t gaps_off, uint64_t gaps_len, uint64_t vals_off, uint64_t vals_len,
                             uint64_t cell_lo, uint64_t cell_hi, uint64_t nnz_lo, uint64_t nnz_hi, uint64_t n_genes,
                             const int64_t *indptr, uint8_t *scratch, uint32_t *out_idx, void *out_val, int to_float,
                             size_t *zstd_error) {
    const uint64_t n = nnz_hi - nnz_lo;
    if (n == 0) return DC_OK;
    uint8_t *gp = scratch;       /* gaps, byte-planar: n low bytes then n high bytes */
    uint8_t *vp = scratch + 2 * n; /* values, byte-planar */

    ZSTD_DCtx *ctx = ZSTD_createDCtx();
    if (ctx == NULL) return DC_NO_MEMORY;
    size_t r = ZSTD_decompressDCtx(ctx, gp, 2 * n, blob + gaps_off, gaps_len);
    if (ZSTD_isError(r)) { *zstd_error = r; ZSTD_freeDCtx(ctx); return DC_ZSTD_ERROR; }
    if (r != 2 * n) { ZSTD_freeDCtx(ctx); return DC_SIZE_MISMATCH; }
    r = ZSTD_decompressDCtx(ctx, vp, 2 * n, blob + vals_off, vals_len);
    ZSTD_freeDCtx(ctx);
    if (ZSTD_isError(r)) { *zstd_error = r; return DC_ZSTD_ERROR; }
    if (r != 2 * n) return DC_SIZE_MISMATCH;

    const uint8_t *glo = gp, *ghi = gp + n, *vlo = vp, *vhi = vp + n;
    for (uint64_t cell = cell_lo; cell < cell_hi; cell++) {
        int64_t s = indptr[cell] - (int64_t)nnz_lo, e = indptr[cell + 1] - (int64_t)nnz_lo;
        if (s < 0 || e < s || (uint64_t)e > n) return DC_BAD_INDPTR;
        uint64_t run = 0; /* the first gap of each cell is the gene index itself */
        for (int64_t j = s; j < e; j++) {
            run += (uint64_t)glo[j] | ((uint64_t)ghi[j] << 8);
            out_idx[nnz_lo + (uint64_t)j] = (uint32_t)run;
        }
        if (e > s && run >= n_genes) return DC_INDEX_OUT_OF_RANGE;
    }
    if (to_float) {
        float *ov = (float *)out_val;
        for (uint64_t j = 0; j < n; j++) ov[nnz_lo + j] = (float)((uint32_t)vlo[j] | ((uint32_t)vhi[j] << 8));
    } else {
        uint32_t *ov = (uint32_t *)out_val;
        for (uint64_t j = 0; j < n; j++) ov[nnz_lo + j] = (uint32_t)vlo[j] | ((uint32_t)vhi[j] << 8);
    }
    return DC_OK;
}

static PyObject *py_decode_chunk(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_buffer blob = {0}, indptr = {0}, scratch = {0}, out_idx = {0}, out_val = {0};
    unsigned long long gaps_off, gaps_len, vals_off, vals_len, cell_lo, cell_hi, nnz_lo, nnz_hi, n_genes, n;
    int to_float;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "y*KKKKKKKKKy*w*w*w*p", &blob, &gaps_off, &gaps_len, &vals_off, &vals_len, &cell_lo, &cell_hi,
                          &nnz_lo, &nnz_hi, &n_genes, &indptr, &scratch, &out_idx, &out_val, &to_float))
        return NULL;

    unsigned long long blob_len = (unsigned long long)blob.len;
    if (gaps_off > blob_len || gaps_len > blob_len - gaps_off || vals_off > blob_len || vals_len > blob_len - vals_off) {
        PyErr_SetString(PyExc_ValueError, "chunk stream lies outside the tile blob");
        goto done;
    }
    if (nnz_hi < nnz_lo || cell_hi < cell_lo) {
        PyErr_SetString(PyExc_ValueError, "invalid chunk range");
        goto done;
    }
    n = nnz_hi - nnz_lo;
    if ((unsigned long long)scratch.len < 4 * n) {
        PyErr_SetString(PyExc_ValueError, "scratch buffer too small (need 4 bytes per nonzero of the chunk)");
        goto done;
    }
    if ((unsigned long long)out_idx.len < 4 * nnz_hi || (unsigned long long)out_val.len < 4 * nnz_hi) {
        PyErr_SetString(PyExc_ValueError, "output buffers must hold at least nnz_hi 4-byte elements");
        goto done;
    }
    if ((unsigned long long)indptr.len < 8 * (cell_hi + 1)) {
        PyErr_SetString(PyExc_ValueError, "indptr too short");
        goto done;
    }
    if (!ALIGNED(indptr.buf, 8) || !ALIGNED(out_idx.buf, 4) || !ALIGNED(out_val.buf, 4)) {
        PyErr_SetString(PyExc_ValueError, "misaligned buffer");
        goto done;
    }
    {
        const int64_t *ip = (const int64_t *)indptr.buf;
        if (ip[cell_lo] != (int64_t)nnz_lo || ip[cell_hi] != (int64_t)nnz_hi) {
            PyErr_SetString(PyExc_ValueError, "indptr does not match the chunk's nonzero range");
            goto done;
        }
        size_t zerr = 0;
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = decode_chunk_impl((const uint8_t *)blob.buf, gaps_off, gaps_len, vals_off, vals_len, cell_lo, cell_hi, nnz_lo, nnz_hi,
                               n_genes, ip, (uint8_t *)scratch.buf, (uint32_t *)out_idx.buf, out_val.buf, to_float, &zerr);
        Py_END_ALLOW_THREADS
        switch (rc) {
        case DC_OK: result = Py_None; Py_INCREF(result); break;
        case DC_ZSTD_ERROR: PyErr_Format(PyExc_ValueError, "corrupt chunk: zstd error: %s", ZSTD_getErrorName(zerr)); break;
        case DC_SIZE_MISMATCH: PyErr_SetString(PyExc_ValueError, "corrupt chunk: decompressed size mismatch"); break;
        case DC_BAD_INDPTR: PyErr_SetString(PyExc_ValueError, "corrupt tile: invalid indptr"); break;
        case DC_INDEX_OUT_OF_RANGE: PyErr_SetString(PyExc_ValueError, "corrupt chunk: gene index out of range"); break;
        default: PyErr_NoMemory();
        }
    }
done:
    PyBuffer_Release(&blob); PyBuffer_Release(&indptr); PyBuffer_Release(&scratch); PyBuffer_Release(&out_idx); PyBuffer_Release(&out_val);
    return result;
}

/* ------------------------------------------------------------------------------------------------ apply_exceptions */

/*
 * Values above 65535 and gaps above 65535 do not fit the uint16 streams. The streams hold 65535 at such entries and the chunk
 * has an exception frame instead: a zstd frame that decompresses to k little-endian uint32 positions (chunk-local nonzero
 * indices, strictly increasing) followed by k uint32 payloads. Run after decode_chunk, only for chunks that have a frame:
 *
 *   kind 0 (values)  payload = the true value (65536 .. INT32_MAX); it replaces the stored value.
 *   kind 1 (gaps)    payload = true gap - 65535 (1 .. n_genes - 1); it is added to the already decoded gene index of that
 *                    nonzero and of every later nonzero of the same cell.
 */

enum { EX_OK = 0, EX_ZSTD_ERROR, EX_BAD_FRAME, EX_BAD_POSITION, EX_BAD_PAYLOAD, EX_BAD_INDPTR, EX_INDEX_OUT_OF_RANGE, EX_NO_MEMORY };

static int apply_exceptions_impl(const uint8_t *frame, uint64_t frame_len, int kind, uint64_t cell_lo, uint64_t cell_hi,
                                 uint64_t nnz_lo, uint64_t nnz_hi, uint64_t n_genes, const int64_t *indptr, uint32_t *out_idx,
                                 void *out_val, int to_float, size_t *zstd_error) {
    const uint64_t n = nnz_hi - nnz_lo;
    unsigned long long size = ZSTD_getFrameContentSize(frame, (size_t)frame_len);
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN || size % 8 != 0 || size / 8 > n) return EX_BAD_FRAME;
    const uint64_t k = size / 8; /* at most one exception per nonzero, which also bounds the allocation below */
    if (k == 0) return EX_OK;
    uint32_t *buf = (uint32_t *)malloc((size_t)size);
    if (buf == NULL) return EX_NO_MEMORY;
    size_t r = ZSTD_decompress(buf, (size_t)size, frame, (size_t)frame_len);
    if (ZSTD_isError(r)) { *zstd_error = r; free(buf); return EX_ZSTD_ERROR; }
    if (r != size) { free(buf); return EX_BAD_FRAME; }
    const uint32_t *pos = buf, *pay = buf + k;
    int rc = EX_OK;
    for (uint64_t i = 0; i < k && rc == EX_OK; i++) {
        if (pos[i] >= n || (i > 0 && pos[i] <= pos[i - 1])) { rc = EX_BAD_POSITION; break; }
        const uint64_t p = nnz_lo + pos[i];
        if (kind == 0) {
            if (pay[i] <= 65535u || pay[i] > 0x7fffffffu) { rc = EX_BAD_PAYLOAD; break; }
            if (to_float) ((float *)out_val)[p] = (float)pay[i];
            else ((uint32_t *)out_val)[p] = pay[i];
        } else {
            if (pay[i] == 0 || pay[i] >= n_genes) { rc = EX_BAD_PAYLOAD; break; }
            uint64_t lo = cell_lo, hi = cell_hi; /* the last cell c in [cell_lo, cell_hi) with indptr[c] <= p */
            while (hi - lo > 1) {
                uint64_t mid = lo + (hi - lo) / 2;
                if ((uint64_t)indptr[mid] <= p) lo = mid; else hi = mid;
            }
            const int64_t end = indptr[lo + 1];
            if ((uint64_t)indptr[lo] > p || (uint64_t)end <= p || (uint64_t)end > nnz_hi) { rc = EX_BAD_INDPTR; break; }
            if ((uint64_t)out_idx[end - 1] + pay[i] >= n_genes) { rc = EX_INDEX_OUT_OF_RANGE; break; } /* the last index is the largest */
            for (int64_t j = (int64_t)p; j < end; j++) out_idx[j] += pay[i];
        }
    }
    free(buf);
    return rc;
}

static PyObject *py_apply_exceptions(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_buffer blob = {0}, indptr = {0}, out_idx = {0}, out_val = {0};
    unsigned long long exc_off, exc_len, cell_lo, cell_hi, nnz_lo, nnz_hi, n_genes;
    int kind, to_float;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "y*KKiKKKKKy*w*w*p", &blob, &exc_off, &exc_len, &kind, &cell_lo, &cell_hi, &nnz_lo, &nnz_hi,
                          &n_genes, &indptr, &out_idx, &out_val, &to_float))
        return NULL;

    unsigned long long blob_len = (unsigned long long)blob.len;
    if (kind != 0 && kind != 1) {
        PyErr_SetString(PyExc_ValueError, "kind must be 0 (values) or 1 (gaps)");
        goto done;
    }
    if (exc_off > blob_len || exc_len > blob_len - exc_off) {
        PyErr_SetString(PyExc_ValueError, "exception frame lies outside the tile blob");
        goto done;
    }
    if (nnz_hi < nnz_lo || cell_hi <= cell_lo) {
        PyErr_SetString(PyExc_ValueError, "invalid chunk range");
        goto done;
    }
    if ((unsigned long long)out_idx.len < 4 * nnz_hi || (unsigned long long)out_val.len < 4 * nnz_hi) {
        PyErr_SetString(PyExc_ValueError, "output buffers must hold at least nnz_hi 4-byte elements");
        goto done;
    }
    if ((unsigned long long)indptr.len < 8 * (cell_hi + 1)) {
        PyErr_SetString(PyExc_ValueError, "indptr too short");
        goto done;
    }
    if (!ALIGNED(indptr.buf, 8) || !ALIGNED(out_idx.buf, 4) || !ALIGNED(out_val.buf, 4)) {
        PyErr_SetString(PyExc_ValueError, "misaligned buffer");
        goto done;
    }
    {
        const int64_t *ip = (const int64_t *)indptr.buf;
        if (ip[cell_lo] != (int64_t)nnz_lo || ip[cell_hi] != (int64_t)nnz_hi) {
            PyErr_SetString(PyExc_ValueError, "indptr does not match the chunk's nonzero range");
            goto done;
        }
        size_t zerr = 0;
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = apply_exceptions_impl((const uint8_t *)blob.buf + exc_off, exc_len, kind, cell_lo, cell_hi, nnz_lo, nnz_hi, n_genes, ip,
                                   (uint32_t *)out_idx.buf, out_val.buf, to_float, &zerr);
        Py_END_ALLOW_THREADS
        switch (rc) {
        case EX_OK: result = Py_None; Py_INCREF(result); break;
        case EX_ZSTD_ERROR: PyErr_Format(PyExc_ValueError, "corrupt exception frame: zstd error: %s", ZSTD_getErrorName(zerr)); break;
        case EX_BAD_FRAME: PyErr_SetString(PyExc_ValueError, "corrupt exception frame: bad size"); break;
        case EX_BAD_POSITION: PyErr_SetString(PyExc_ValueError, "corrupt exception frame: positions are not increasing or are out of range"); break;
        case EX_BAD_PAYLOAD: PyErr_SetString(PyExc_ValueError, "corrupt exception frame: value out of range"); break;
        case EX_BAD_INDPTR: PyErr_SetString(PyExc_ValueError, "corrupt tile: invalid indptr"); break;
        case EX_INDEX_OUT_OF_RANGE: PyErr_SetString(PyExc_ValueError, "corrupt chunk: gene index out of range"); break;
        default: PyErr_NoMemory();
        }
    }
done:
    PyBuffer_Release(&blob); PyBuffer_Release(&indptr); PyBuffer_Release(&out_idx); PyBuffer_Release(&out_val);
    return result;
}

/* ------------------------------------------------------------------------------------------------ gather_rows */

static int gather_rows_impl(const int64_t *src_indptr, const uint32_t *src_idx, const uint32_t *src_val, const int64_t *rows, uint64_t n_rows,
                            uint64_t n_src_rows, const int64_t *out_offsets, uint64_t out_len, uint32_t *out_idx, void *out_val, int convert) {
    for (uint64_t i = 0; i < n_rows; i++) {
        int64_t r = rows[i];
        if (r < 0 || (uint64_t)r >= n_src_rows) return 1;
        int64_t a = src_indptr[r], b = src_indptr[r + 1], o = out_offsets[i];
        int64_t len = b - a;
        if (len < 0 || o < 0 || (uint64_t)(o + len) > out_len) return 2;
        memcpy(out_idx + o, src_idx + a, (size_t)len * sizeof(uint32_t));
        if (!convert) {
            memcpy((uint32_t *)out_val + o, src_val + a, (size_t)len * sizeof(uint32_t));
        } else {
            float *ov = (float *)out_val + o;
            const uint32_t *sv = src_val + a;
            for (int64_t j = 0; j < len; j++) ov[j] = (float)sv[j];
        }
    }
    return 0;
}

static PyObject *py_gather_rows(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_buffer src_indptr = {0}, src_idx = {0}, src_val = {0}, rows = {0}, out_offsets = {0}, out_idx = {0}, out_val = {0};
    int convert;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*w*w*p", &src_indptr, &src_idx, &src_val, &rows, &out_offsets, &out_idx, &out_val, &convert)) return NULL;
    if (!ALIGNED(src_indptr.buf, 8) || !ALIGNED(rows.buf, 8) || !ALIGNED(out_offsets.buf, 8) || !ALIGNED(src_idx.buf, 4) ||
        !ALIGNED(src_val.buf, 4) || !ALIGNED(out_idx.buf, 4) || !ALIGNED(out_val.buf, 4)) {
        PyErr_SetString(PyExc_ValueError, "misaligned buffer");
        goto done;
    }
    if (src_indptr.len < 8 || out_offsets.len < rows.len || out_idx.len != out_val.len) {
        PyErr_SetString(PyExc_ValueError, "inconsistent buffer sizes");
        goto done;
    }
    {
        uint64_t n_rows = (uint64_t)rows.len / 8, n_src_rows = (uint64_t)src_indptr.len / 8 - 1, out_len = (uint64_t)out_idx.len / 4;
        const int64_t *ip = (const int64_t *)src_indptr.buf;
        if ((uint64_t)ip[n_src_rows] * 4 > (uint64_t)src_idx.len || (uint64_t)ip[n_src_rows] * 4 > (uint64_t)src_val.len) {
            PyErr_SetString(PyExc_ValueError, "source arrays shorter than indptr[-1]");
            goto done;
        }
        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = gather_rows_impl(ip, (const uint32_t *)src_idx.buf, (const uint32_t *)src_val.buf, (const int64_t *)rows.buf, n_rows, n_src_rows,
                              (const int64_t *)out_offsets.buf, out_len, (uint32_t *)out_idx.buf, out_val.buf, convert);
        Py_END_ALLOW_THREADS
        if (rc == 1) PyErr_SetString(PyExc_IndexError, "row index out of range");
        else if (rc == 2) PyErr_SetString(PyExc_ValueError, "output buffers too small for the requested rows");
        else { result = Py_None; Py_INCREF(result); }
    }
done:
    PyBuffer_Release(&src_indptr); PyBuffer_Release(&src_idx); PyBuffer_Release(&src_val); PyBuffer_Release(&rows);
    PyBuffer_Release(&out_offsets); PyBuffer_Release(&out_idx); PyBuffer_Release(&out_val);
    return result;
}

/* ------------------------------------------------------------------------------------------------ compress_stream */

static PyObject *py_compress_bound(PyObject *Py_UNUSED(self), PyObject *args) {
    unsigned long long n;
    if (!PyArg_ParseTuple(args, "K", &n)) return NULL;
    return PyLong_FromUnsignedLongLong((unsigned long long)ZSTD_compressBound(2 * (size_t)n));
}

/* compress_stream(src_uint16_array, level, out_buffer) -> compressed size. The array is split into low and high byte planes. */
static PyObject *py_compress_stream(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_buffer src = {0}, out = {0};
    int level;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "y*iw*", &src, &level, &out)) return NULL;
    if (src.len % 2 != 0) { PyErr_SetString(PyExc_ValueError, "source must be a uint16 array"); goto done; }
    if (level < ZSTD_minCLevel() || level > ZSTD_maxCLevel()) { PyErr_SetString(PyExc_ValueError, "invalid zstd compression level"); goto done; }
    {
        size_t n = (size_t)src.len / 2;
        if ((size_t)out.len < ZSTD_compressBound(2 * n)) { PyErr_SetString(PyExc_ValueError, "output buffer smaller than compress_bound"); goto done; }
        uint8_t *planar = (uint8_t *)malloc(2 * n > 0 ? 2 * n : 1);
        if (planar == NULL) { PyErr_NoMemory(); goto done; }
        size_t csize;
        Py_BEGIN_ALLOW_THREADS
        for (size_t j = 0; j < n; j++) { /* memcpy: the source array need not be 2-byte aligned */
            uint16_t v;
            memcpy(&v, (const uint8_t *)src.buf + 2 * j, 2);
            planar[j] = (uint8_t)(v & 0xff);
            planar[n + j] = (uint8_t)(v >> 8);
        }
        csize = ZSTD_compress(out.buf, (size_t)out.len, planar, 2 * n, level);
        free(planar);
        Py_END_ALLOW_THREADS
        if (ZSTD_isError(csize)) PyErr_Format(PyExc_RuntimeError, "zstd compression failed: %s", ZSTD_getErrorName(csize));
        else result = PyLong_FromUnsignedLongLong((unsigned long long)csize);
    }
done:
    PyBuffer_Release(&src); PyBuffer_Release(&out);
    return result;
}

/* compress_bytes(src, level) -> bytes: one plain zstd frame (with its content size), for the exception frames. */
static PyObject *py_compress_bytes(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_buffer src = {0};
    int level;
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "y*i", &src, &level)) return NULL;
    if (level < ZSTD_minCLevel() || level > ZSTD_maxCLevel()) { PyErr_SetString(PyExc_ValueError, "invalid zstd compression level"); goto done; }
    {
        size_t cap = ZSTD_compressBound((size_t)src.len);
        PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)cap);
        if (out == NULL) goto done;
        size_t csize;
        Py_BEGIN_ALLOW_THREADS
        csize = ZSTD_compress(PyBytes_AS_STRING(out), cap, src.buf, (size_t)src.len, level);
        Py_END_ALLOW_THREADS
        if (ZSTD_isError(csize)) {
            PyErr_Format(PyExc_RuntimeError, "zstd compression failed: %s", ZSTD_getErrorName(csize));
            Py_DECREF(out);
        } else if (_PyBytes_Resize(&out, (Py_ssize_t)csize) == 0) {
            result = out;
        }
    }
done:
    PyBuffer_Release(&src);
    return result;
}

/* ------------------------------------------------------------------------------------------------ module */

static PyMethodDef methods[] = {
    {"decode_chunk", py_decode_chunk, METH_VARARGS,
     "decode_chunk(blob, gaps_off, gaps_len, vals_off, vals_len, cell_lo, cell_hi, nnz_lo, nnz_hi, n_genes, indptr, scratch, out_indices, out_values, to_float)"},
    {"apply_exceptions", py_apply_exceptions, METH_VARARGS,
     "apply_exceptions(blob, exc_off, exc_len, kind, cell_lo, cell_hi, nnz_lo, nnz_hi, n_genes, indptr, out_indices, out_values, to_float)"},
    {"compress_bytes", py_compress_bytes, METH_VARARGS, "compress_bytes(bytes_like, level) -> bytes (one zstd frame)"},
    {"gather_rows", py_gather_rows, METH_VARARGS,
     "gather_rows(src_indptr, src_indices, src_values, rows, out_offsets, out_indices, out_values, convert_to_float)"},
    {"compress_bound", py_compress_bound, METH_VARARGS, "compress_bound(n_elements) -> max compressed size of a uint16 stream"},
    {"compress_stream", py_compress_stream, METH_VARARGS, "compress_stream(uint16_array, level, out_buffer) -> compressed size"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_core", "deltacells C core", -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddStringConstant(m, "ZSTD_VERSION", ZSTD_versionString()) < 0) { Py_DECREF(m); return NULL; }
    return m;
}
