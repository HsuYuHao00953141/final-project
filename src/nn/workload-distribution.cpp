#include "nn-quants.hpp"
#include "nn-core.hpp"
#include "workload-distribution.hpp"

NnUint distribute_mode = 0;
NnUint nSlice = 1;  // 預設值

NnUint yqSlice_dim = 0;
NnUint logitsSlice_dim = 0;
// row dim: n, col dim: d

//distributer

/*--------------------------------sliceKvCache distribute------------------------------------
NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes) {
    NnKvCacheSlice s;
    assert(kvDim % nNodes == 0);
    s.kvDim0 = kvDim / nNodes;
    s.keySize = size2D(F_32, seqLen, s.kvDim0);
    s.valueSize = size2D(F_32, seqLen, s.kvDim0);
    return s;
}
---------------------------------------------------------------------------------------*/

NnKvCacheSlice sliceKvCache_distribute(NnUint partition, NnUint seqLen, const NnKvCacheSlice &slice) {
    NnKvCacheSlice w;
    //assert(kvDim % nNodes == 0);
    w.kvDim0 = slice.kvDim0 * partition;
    w.keySize = size2D(F_32, seqLen, w.kvDim0);
    w.valueSize = size2D(F_32, seqLen, w.kvDim0);
    return w;
}

/*--------------------------------sliceKvCache distribute------------------------------------
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches) {
    NnMultiHeadAttSlice s;
    assert(nHeads % nNodes == 0);
    s.nHeads = nHeads;
    s.nHeads0 = nHeads / nNodes;
    s.attSize = size2D(F_32, nBatches, s.nHeads0 * seqLen);
    return s;
}
---------------------------------------------------------------------------------------*/
NnMultiHeadAttSlice sliceMultiHeadAtt_distribute(NnUint nHeads, NnUint seqLen, NnUint partition, NnUint nBatches, const NnMultiHeadAttSlice &slice) {
    NnMultiHeadAttSlice w;
    //assert(nHeads % nNodes == 0);
    w.nHeads = slice.nHeads;
    w.nHeads0 = slice.nHeads0 * partition;
    w.attSize = size2D(F_32, nBatches, w.nHeads0 * seqLen);
    return w;
}

/*--------------------------------workload_rowslicedistribute------------------------------------
modify from slicers function
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    //assert(d % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.d = d;
    s.d0 = d / nNodes;
    s.n = n;
    s.size = size2D(type, s.n, d);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}
---------------------------------------------------------------------------------------*/

NnRowMatmulSlice workload_rowslicedistribute(NnUint partition, const NnRowMatmulSlice &slice) {
    NnRowMatmulSlice w;
    //assert(d % nNodes == 0);

    w.type = slice.type;
    w.n = slice.n;
    w.d = slice.d;
    w.d0 = slice.d0 * partition;
    w.size = size2D(w.type, w.n, w.d);
    w.sliceSize = size2D(w.type, w.n, w.d0);

    return w;
}

/*--------------------------------workload_colslicedistribute------------------------------------
modify from slicers function
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);

NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnColMatmulSlice s;
    assert(n % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.n = n;
    s.n0 = n / nNodes;
    s.d = d;
    s.size = size2D(type, n, d);
    s.sliceSize = size2D(type, s.n0, d);
    return s;
}
---------------------------------------------------------------------------------------*/
NnColMatmulSlice workload_colslicedistribute(NnUint partition, const NnColMatmulSlice &slice) {
    NnColMatmulSlice w;
    //assert(d % nNodes == 0);

    w.type = slice.type;
    w.n = slice.n;
    w.n0 = slice.n0 * partition;
    w.d = slice.d;
    w.size = size2D(w.type, w.n, w.d);
    w.sliceSize = size2D(w.type, w.n0, w.d);

    return w;
}

NnRopeSlice sliceRope_distribute(NnUint dim, NnUint kvDim, NnUint nKvHeads, NnUint seqLen, NnUint headSize, float ropeTheta, NnUint sliceOffset, NnUint sliceCount, NnUint nSlice)
{
    NnRopeSlice s;

    s.qDim0 = dim / nSlice;   // 每一單位 Q 維度
    s.kvDim0 = kvDim / nSlice;

    s.qDimStart = s.qDim0 * sliceOffset;
    s.qDimEnd   = s.qDimStart + s.qDim0 * sliceCount;
    s.kvDimStart = s.kvDim0 * sliceOffset;

    s.qShift = s.qDimStart - s.kvDimStart;
    s.sliceDim = s.qDimEnd - s.kvDimStart;

    assert(s.qDim0 % 2 == 0);
    assert(s.kvDim0 % 2 == 0);
    assert(s.sliceDim % 2 == 0);

    s.kvDim = kvDim;
    s.nKvHeads = nKvHeads;
    s.seqLen = seqLen;
    s.headSize = headSize;
    s.ropeTheta = ropeTheta;
    s.cacheSize = size2D(F_32, s.seqLen, s.sliceDim);

    return s;
}