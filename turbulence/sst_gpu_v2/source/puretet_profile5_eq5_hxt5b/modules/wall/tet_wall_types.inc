struct TetWallMatchPoint{
  std::int32_t cell=-1;
  std::int32_t g[8]={0};
  double b[8]={0};
  double y=0;
};

// Stage 2.B.2 device-only ray record: same FP64 barycentric values,
// global velocity DOFs and containing tet as original full record.
// BF3 enrichment shape values are deterministic cubic polynomials of l[4].
// Host ray tracing/validation CONTINUES to use the full TetWallMatchPoint.
struct TetWallPackedPoint{
  std::int32_t cell=-1;
  std::int32_t g[8]={0};
  double l[4]={0};
  double y=0; // Certified physical ray height, possibly shortened locally.
};
static_assert(sizeof(TetWallPackedPoint)==80,
              "Stage27 adaptive ray record must contain exactly 80 bytes");

struct TetWallQuadCache{
  double bw[8]; // exact P1+BF3 wall trace
  double bp[8]; // in-wall-adjacent-tet normal-column Picard functional
  double w=0.0,ybar=0.0;
};

struct TetFacePlan{
  std::int32_t cell=0;
  std::int8_t lf=0;
  double area=0,n[3]={0,0,0};
};

