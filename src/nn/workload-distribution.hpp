#include "nn-quants.hpp"
#include "nn-core.hpp"
#include <cassert>

//1: act, 0: dact
extern NnUint distribute_mode;

//number of slicer's slice
extern NnUint nSlice;

//rope slice index
//extern NnUint index;

//yqSlice dim
extern NnUint yqSlice_dim;

//logitsSlice dim
extern NnUint logitsSlice_dim;

// row dim: n, col dim: d

/*--------------------------------workload partition------------------------------------
include from NnRowMatmulSlice struct


---------------------------------------------------------------------------------------*/


/*--------------------------------workload partition------------------------------------
include from NnColMatmulSlice struct

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint n;
    NnUint n0;
    NnUint d;
    NnSize2D size;
    NnSize2D sliceSize;
} NnColMatmulSlice;
-------------------------------------------------------------------------------------------*/

/*--------------------------------sliceKvCache distribute------------------------------------
modify from slicers function
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    //assert(d % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.d0 = d / nNodes;
    s.n = n;
    s.size = size2D(type, s.n, d);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}
------------------------------------------------------------------------------------------------*/
NnKvCacheSlice sliceKvCache_distribute(NnUint partition, NnUint seqLen, const NnKvCacheSlice &slice);

NnMultiHeadAttSlice sliceMultiHeadAtt_distribute(NnUint nHeads, NnUint seqLen, NnUint partition, NnUint nBatches, const NnMultiHeadAttSlice &slice);

/*--------------------------------workload_rowslicedistribute------------------------------------
modify from slicers function
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    //assert(d % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.d0 = d / nNodes;
    s.n = n;
    s.size = size2D(type, s.n, d);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}
------------------------------------------------------------------------------------------------*/
NnRowMatmulSlice workload_rowslicedistribute(NnUint partition, const NnRowMatmulSlice &slice);

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
----------------------------------------------------------------------------------------------*/
NnColMatmulSlice workload_colslicedistribute(NnUint partition, const NnColMatmulSlice &slice);

NnRopeSlice sliceRope_distribute(NnUint dim, NnUint kvDim, NnUint nKvHeads, NnUint seqLen, NnUint headSize, float ropeTheta, NnUint sliceOffset, NnUint sliceCount, NnUint nSlice);

