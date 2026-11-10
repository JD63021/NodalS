// Stage22D.4D: matrix-free coupled vector momentum comparison.
// The three scalar diagonal blocks are exactly those assembled in Stage22D.4C.
// Off-diagonal frozen face action uses exactly the same local nu_eff and beta
// computed during that outer iteration's Nitsche assembly. No model change.
// The pressure Schur approximation remains component diagonal (CONTROLLED).
// This header is included INSIDE namespace hxt5d.
#include "bazilevs_b_tangent.inc"
#include "generalized_robin_tangent.inc"

__global__ void s22d4d_offblock_action(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const double*coeff,int nv,const Real*x,Real*y,double gammaN,
    bool bf3FluxMode,bool sampledConductance){
  const int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto face=F[fi];
  const auto geom=tg[face.cell];
  const auto vel=tv[face.cell];
  const double h=3.*geom.volume/face.area;
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    const std::size_t iq=(std::size_t)fi*12u+(std::size_t)q;
    const double nueff=coeff[2u*iq],beta=coeff[2u*iq+1u];
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
    double ph[8],gr[8][3],gp[8][3],dn[8];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,geom,gp);
    double uq[3]={},upic[3]={},dnu[3]={};
    for(int b=0;b<8;++b){
      const int ib=vel.g[b];
      dn[b]=gp[b][0]*n[0]+gp[b][1]*n[1]+gp[b][2]*n[2];
      for(int j=0;j<3;++j){
        const double v=(double)x[(std::size_t)j*(std::size_t)nv+(std::size_t)ib];
        uq[j]+=ca.bw[b]*v;upic[j]+=ca.bp[b]*v;dnu[j]+=dn[b]*v;
      }
    }
    for(int a=0;a<8;++a){
      const int ia=vel.g[a];const double pa=ca.bw[a],dna=dn[a];
      for(int i=0;i<3;++i){
        double cross=0.;
        for(int j=0;j<3;++j){
          if(i==j)continue;
          const double nij=n[i]*n[j];
          // This is exactly the cross-component part of the assembled
          // diagonal-block tensor: normal terms use the wall trace,
          // tangential friction optionally uses the sampled column basis.
          const double normalContribution=bf3FluxMode ?
            gammaN*nueff/h*pa*uq[j] :
            -nueff*pa*dnu[j]-nueff*dna*uq[j]+gammaN*nueff/h*pa*uq[j];
          const double tangentialContribution=beta*(sampledConductance ?
            ca.bp[a]*upic[j] : pa*uq[j]);
          cross+=nij*(normalContribution-tangentialContribution);
        }
        atomicAdd(y+(std::size_t)i*(std::size_t)nv+(std::size_t)ia,(Real)(ca.w*cross));
      }
    }
  }
}

// F1B: freeze the scalar off-component coupling for each wall face once
// after momentum wall assembly. All 12 original quadrature samples, nu_eff,
// beta, BF3 normal term and SC1 sampled-basis option are preserved.
// Storage layout: C[face*64+a*8+b] = integral( normal_stiffness - robin ).
// Off-component tensor A_ij = n_i*n_j*C_ab, i != j.
__global__ void f1b_build_face_coupling(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetGeom*tg,
    const double*coeff,double*faceC,double gammaN,
    bool bf3FluxMode,bool sampledConductance){
  const int fi=(int)blockIdx.x;
  const int pair=(int)threadIdx.x;
  if(fi>=nf||pair>=64)return;
  const auto face=F[fi];
  const auto geom=tg[face.cell];
  const double h=3.*geom.volume/face.area;
  const int a=pair/8,b=pair%8;
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  double total=0.;
  for(int q=0;q<12;++q){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    const std::size_t iq=(std::size_t)fi*12u+(std::size_t)q;
    const double nueff=coeff[2u*iq],beta=coeff[2u*iq+1u];
    const double pa=ca.bw[a],pb=ca.bw[b];
    double normal=gammaN*nueff/h*pa*pb;
    if(!bf3FluxMode){
      double lam[4]={0.,0.,0.,0.};
      lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
      lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
      lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
      double ph[8],gr[8][3],gp[8][3];
      hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
      hxt2::tet_phys_grad(gr,geom,gp);
      const double dna=gp[a][0]*n[0]+gp[a][1]*n[1]+gp[a][2]*n[2];
      const double dnb=gp[b][0]*n[0]+gp[b][1]*n[1]+gp[b][2]*n[2];
      normal+=-nueff*(pa*dnb+dna*pb);
    }
    const double robin=beta*(sampledConductance?ca.bp[a]*ca.bp[b]:pa*pb);
    total+=ca.w*(normal-robin);
  }
  faceC[(std::size_t)fi*64u+(std::size_t)pair]=total;
}

// One warp handles a physical wall face: 8 active lanes (one per local test
// basis).  No 12-point quadrature or geometry-gradient reconstruction occurs
// in the repeated FGMRES action. Diagonal CSR, T/Tt and fixed masks remain
// exactly the same as the legacy action.
__global__ void f1b_offblock_cached(
    const hxt5b::TetFacePlan*F,int nf,
    const nodals_hxt1::TetVel*tv,const double*faceC,
    int nv,const Real*x,Real*y){
  const int fi=(int)blockIdx.x;
  const int a=(int)threadIdx.x;
  if(fi>=nf||a>=8)return;
  const auto face=F[fi];
  const auto vel=tv[face.cell];
  const double n0=face.n[0],n1=face.n[1],n2=face.n[2];
  const double*row=faceC+(std::size_t)fi*64u+(std::size_t)a*8u;
  double sums[3]={0.,0.,0.};
  for(int b=0;b<8;++b){
    const int ib=vel.g[b];
    const double c=row[b];
    const double x0=(double)x[ib];
    const double x1=(double)x[(std::size_t)nv+(std::size_t)ib];
    const double x2=(double)x[(std::size_t)2*nv+(std::size_t)ib];
    sums[0]+=c*(n1*x1+n2*x2);
    sums[1]+=c*(n0*x0+n2*x2);
    sums[2]+=c*(n0*x0+n1*x1);
  }
  const int ia=vel.g[a];
  atomicAdd(y+ia,(Real)(n0*sums[0]));
  atomicAdd(y+(std::size_t)nv+ia,(Real)(n1*sums[1]));
  atomicAdd(y+(std::size_t)2*nv+ia,(Real)(n2*sums[2]));
}

__global__ void f1b_difference(int n,const Real*a,const Real*b,Real*d){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)d[i]=a[i]-b[i];
}

__global__ void s22d4d_pack(int nv,const Real*u0,const Real*u1,const Real*u2,
    const Real*b0,const Real*b1,const Real*b2,
    const Real*lag0,const Real*lag1,const Real*lag2,
    const unsigned char*maskXY,const unsigned char*maskZ,
    Real*x,Real*b){
  int a=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(a>=nv)return;
  const Real*u[3]={u0,u1,u2};const Real*rhs[3]={b0,b1,b2};
  const Real*lag[3]={lag0,lag1,lag2};
  for(int i=0;i<3;++i){
    const int ix=i*nv+a;
    const bool fixed=(i==2?maskZ[a]:maskXY[a]);
    x[ix]=fixed?Real(0):u[i][a];
    // Stage22D.4C rhs includes frozen off-block action: subtract it here.
    b[ix]=fixed?Real(0):rhs[i][a]-lag[i][a];
  }
}
__global__ void s22d4d_unpack(int nv,const Real*x,Real*u0,Real*u1,Real*u2){
  int a=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(a>=nv)return;
  u0[a]=x[a];u1[a]=x[(std::size_t)nv+a];u2[a]=x[(std::size_t)2*nv+a];
}
__global__ void s22d4d_residual(int nv,const Real*b,const Real*ax,
    const unsigned char*xy,const unsigned char*z,Real*r){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<3*nv){const int j=i/nv,a=i%nv;
    r[i]=((j==2?z[a]:xy[a]))?Real(0):b[i]-ax[i];
  }
}
__global__ void s22d4d_zero_fixed(int nv,
    const unsigned char*xy,const unsigned char*z,Real*x){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<3*nv){const int j=i/nv,a=i%nv;
    if(j==2?z[a]:xy[a])x[i]=Real(0);
  }
}
// NODALS_IMPLICIT_DEVIATORIC_VOLUME_V1: new optional cross-component
// volume contribution, frozen per SIMPLE/Picard outer iteration.
struct S22D4DOperator {
  hxt4b::GpuCSR&A; const hxt4b::ColoringHost&C;
  const Real*v0,*v1,*v2;
  const unsigned char*xy,*fixed;
  const hxt5b::TetFacePlan*faces;
  const hxt5b::TetWallQuadCache*quads;
  const nodals_hxt1::TetVel*tvel;
  const nodals_hxt1::TetGeom*tgeom;
  const double*coeff;
  int nfaces; double gamma;
  Real* bf3Scratch; bool bf3FluxMode; bool sampledConductance=false;
  const double*f1bFaceCoupling=nullptr;
  // Bazilevs-B: additional symmetric tangential consistency operator.
  // 0 off, 1 molecular nu, 2 frozen nu + nut.
  int bazbViscosityMode=0;
  double bazbMolecularNu=0.0;
  void set_bazb(int mode,double nu){bazbViscosityMode=mode;bazbMolecularNu=nu;}
  // Generalized Robin-Nitsche: physical F2 beta remains the baseline matrix.
  int genrobinViscosityMode=0;
  double genrobinMolecularNu=0.0,genrobinC=15.0;
  void set_genrobin(int mode,double nu,double c){
    genrobinViscosityMode=mode;genrobinMolecularNu=nu;genrobinC=c;
  }
  // Physical volume tensor; default OFF leaves the old operator byte-for-byte.
  // NODALS_EMAC_PICARD_V1: frozen-velocity cross-component convection tensor.
  bool emacPicard=false;
  const Real *emacU0=nullptr,*emacU1=nullptr,*emacU2=nullptr;
  std::uint64_t emacNTet=0;
  void set_emac_picard(bool enabled,const Real*u0,const Real*u1,
      const Real*u2,std::uint64_t nc){
    emacPicard=enabled;
    if(enabled){emacU0=u0;emacU1=u1;emacU2=u2;emacNTet=nc;}
  }
  bool implicitDev=false;
  const nodals_hxt1::Point*devPoints=nullptr;
  const nodals_hxt1::TetConn*devTetConn=nullptr;
  const nodals_hxt1::TetVel*devTetVel=nullptr;
  const nodals_hxt1::TetGeom*devTetGeom=nullptr;
  const Real *devU0=nullptr,*devU1=nullptr,*devU2=nullptr,*devK=nullptr,*devOmega=nullptr;
  std::uint64_t devNTet=0;
  double devRadius=0.,devNu=0.,devKFloor=0.,devOmegaFloor=0.;
  void set_implicit_dev(bool on,
      const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
      const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
      const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
      std::uint64_t nc,double radius,double nu,double kFloor,double omegaFloor){
    implicitDev=on;
    if(on){devPoints=pts;devTetConn=tc;devTetVel=tv;devTetGeom=tg;
      devU0=u0;devU1=u1;devU2=u2;devK=k;devOmega=omega;
      devNTet=nc;devRadius=radius;devNu=nu;
      devKFloor=kFloor;devOmegaFloor=omegaFloor;}
  }
  void apply(const Real*x,Real*y)const {
    const int nv=A.n;
    const Real* physical=x;
    if(bf3FluxMode){
      HXT1_CUDA(cudaMemcpy(bf3Scratch,x,(std::size_t)3*(std::size_t)nv*sizeof(Real),cudaMemcpyDeviceToDevice));
      s22d4i_T_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tvel,
        bf3Scratch,bf3Scratch+nv,bf3Scratch+2*(std::size_t)nv);
      physical=bf3Scratch;
    }
    hxt5b::spmv<<<(nv+255)/256,256>>>(nv,A.row.p,A.col.p,v0,physical,y);
    hxt5b::spmv<<<(nv+255)/256,256>>>(nv,A.row.p,A.col.p,v1,physical+nv,y+nv);
    hxt5b::spmv<<<(nv+255)/256,256>>>(nv,A.row.p,A.col.p,v2,physical+2*(std::size_t)nv,y+2*(std::size_t)nv);
    if(f1bFaceCoupling)
      f1b_offblock_cached<<<nfaces,32>>>(faces,nfaces,tvel,f1bFaceCoupling,
        nv,physical,y);
    else
      s22d4d_offblock_action<<<nfaces,64>>>(faces,nfaces,quads,tvel,tgeom,
         coeff,nv,physical,y,gamma,bf3FluxMode,sampledConductance);
    if(bazbViscosityMode)
      bazb_tangent_action<<<nfaces,32>>>(faces,nfaces,quads,tvel,tgeom,
         coeff,bazbMolecularNu,bazbViscosityMode,nv,physical,y);
    if(genrobinViscosityMode)
      genrobin_tangent_action<<<nfaces,32>>>(
        faces,nfaces,quads,tvel,tgeom,coeff,genrobinMolecularNu,
        genrobinViscosityMode,genrobinC,nv,physical,y);
    if(emacPicard)
      emac_picard_tet_apply<<<emacNTet,64>>>(
        tvel,tgeom,emacU0,emacU1,emacU2,emacNTet,nv,physical,y);
    if(implicitDev)
      implicit_dev_tet_apply<<<devNTet,64>>>(devPoints,devTetConn,devTetVel,devTetGeom,
        devU0,devU1,devU2,devK,devOmega,devNTet,
        devRadius,devNu,devKFloor,devOmegaFloor,nv,physical,y);
    if(bf3FluxMode)
      s22d4i_Tt_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tvel,y,y+nv,y+2*(std::size_t)nv);
    s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,xy,fixed,y);
    HXT1_CUDA(cudaGetLastError());
  }
  void precondition(const Real*rhs,Real*z,int sweeps,double omega)const {
    const int nv=A.n;
    HXT1_CUDA(cudaMemset(z,0,(std::size_t)3*(std::size_t)nv*sizeof(Real)));
    for(int s=0;s<sweeps;++s){
      hxt5b::sweep(A,C,v0,rhs,z,omega,true,xy);
      hxt5b::sweep(A,C,v1,rhs+nv,z+nv,omega,true,xy);
      hxt5b::sweep(A,C,v2,rhs+2*(std::size_t)nv,z+2*(std::size_t)nv,omega,true,fixed);
    }
    s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,xy,fixed,z);
    if(bf3FluxMode)
      s22d4i_Q_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tvel,z,z+nv,z+2*(std::size_t)nv);
    HXT1_CUDA(cudaGetLastError());
  }
};
inline double s22d4d_residual_norm(const S22D4DOperator&op,
   const Real*b,const Real*x,hxt5e::Workspace&W,hxt4a::Reducer<Real>&red){
  const int n=op.A.n;
  op.apply(x,W.w);
  s22d4d_residual<<<(3*n+255)/256,256>>>(n,b,W.w,op.xy,op.fixed,W.r);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(W.r);
}
struct S22D4DResult {int its=0;int cycles=0;double initial=0,final=0,rel=0;bool ok=false;};
// Stage27/5.B.1: exact BF3-projected component FGMRES (Approach A).
// The reduced scalar diagonal block is E_i^T [Tt A T] E_i, NOT the
// unprojected scalar CSR A_ii. This is matrix-free and exact as an action.
// Q E_i delta is the physical parameter-space subspace correction. Since
// T Q = T, the correction has the same effect on the constrained operator.
// The original coupled block FGMRES remains unmodified and selectable.
__global__ void s05b1_embed(int n,int comp,const Real*v,Real*out){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)out[(std::size_t)comp*(std::size_t)n+(std::size_t)i]=v[i];
}
__global__ void s05b1_add(int n,int comp,const Real*v,Real*out){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)out[(std::size_t)comp*(std::size_t)n+(std::size_t)i]+=v[i];
}
inline double s05b1_full_residual(const S22D4DOperator&op,
    const Real*b,const Real*x,Real*ax,Real*r,
    hxt4a::Reducer<Real>&red,bool doNorm){
  int nv=op.A.n;
  op.apply(x,ax);
  s22d4d_residual<<<(3*nv+255)/256,256>>>(nv,b,ax,op.xy,op.fixed,r);
  HXT1_CUDA(cudaGetLastError());
  return doNorm?red.norm(r):std::numeric_limits<double>::quiet_NaN();
}
// Stage 27 / Stage5.B.2: exact diagonal scalar action with one full CSR SpMV.
// Every other component needs its diagonal CSR operator only at *wall BF3 bubble rows*,
// since T^t couples components through the normal residual at those bubbles.
// The complete cross-component wall face action and T/T^t transforms are unchanged.
// Wall BF3 bubble dofs are unique to their boundary face on the supported tet mesh.
__global__ void s05b2_bubble_rows_spmv(int nf,
    const hxt5b::TetFacePlan*faces,const nodals_hxt1::TetVel*tv,
    const std::int64_t*row,const std::int32_t*col,
    const Real*values,const Real*physical,Real*output){
  int fi=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(fi>=nf)return;
  const auto face=faces[fi];
  const int b=tv[face.cell].g[4+(int)face.lf];
  Real sum=Real(0);
  for(std::int64_t j=row[b];j<row[b+1];++j)
    sum+=values[j]*physical[col[j]];
  output[b]=sum;
}
inline bool s05b2_fast_enabled(){
  static const bool fast=[](){
    const char*env=std::getenv("STAGE27_S05B2_FAST_SCALAR_ACTION");
    const bool enabled=(env && std::string(env)=="1");
    if(env && std::string(env)!="0" && std::string(env)!="1")
      throw std::runtime_error("Stage5.B.2 FAST_SCALAR_ACTION must be 0 or 1");
    std::printf("STAGE27_S05B2_COMPONENT_ACTION mode=%s "
      "fullCSRspmvPerComponent=1 otherComponents=BF3_BUBBLE_ROWS_ONLY "
      "offblock=FULL_EXACT Tt=UNCHANGED storage=THREE_CSR_VALUES_UNCHANGED "
      "stage5b1Reference=SELECTABLE status=PASS\n",enabled?"optimized":"reference");
    return enabled;
  }();
  return fast;
}
inline void s05b2_component_action_fast(const S22D4DOperator&op,int comp,
    const Real*embed,Real*out){
  const int nv=op.A.n;
  const Real*physical=embed;
  if(op.bf3FluxMode){
    HXT1_CUDA(cudaMemcpy(op.bf3Scratch,embed,(std::size_t)3*nv*sizeof(Real),cudaMemcpyDeviceToDevice));
    s22d4i_T_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
      op.bf3Scratch,op.bf3Scratch+nv,op.bf3Scratch+2*(std::size_t)nv);
    physical=op.bf3Scratch;
  }
  HXT1_CUDA(cudaMemset(out,0,(std::size_t)3*nv*sizeof(Real)));
  const Real*vals[3]={op.v0,op.v1,op.v2};
  // Fully evaluate the one requested component, including all non-wall rows.
  hxt5b::spmv<<<(nv+255)/256,256>>>(nv,op.A.row.p,op.A.col.p,
    vals[comp],physical+(std::size_t)comp*nv,out+(std::size_t)comp*nv);
  // For other components, T^t needs the diagonal CSR action ONLY at the
  // distinct wall bubble dofs.  Vertex and interior rows in these other
  // components cannot contribute to the extracted component result.
  if(op.bf3FluxMode){
    for(int j=0;j<3;++j){
      if(j==comp)continue;
      s05b2_bubble_rows_spmv<<<(op.nfaces+255)/256,256>>>(op.nfaces,
        op.faces,op.tvel,op.A.row.p,op.A.col.p,vals[j],
        physical+(std::size_t)j*nv,out+(std::size_t)j*nv);
    }
  }
  // Keep all cross-component FE wall actions in the original numerical order.
  if(op.f1bFaceCoupling)
    f1b_offblock_cached<<<op.nfaces,32>>>(op.faces,op.nfaces,op.tvel,
      op.f1bFaceCoupling,nv,physical,out);
  else
    s22d4d_offblock_action<<<op.nfaces,64>>>(op.faces,op.nfaces,op.quads,
      op.tvel,op.tgeom,op.coeff,nv,physical,out,op.gamma,op.bf3FluxMode,op.sampledConductance);
  if(op.bf3FluxMode)
    s22d4i_Tt_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
      out,out+nv,out+2*(std::size_t)nv);
  s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,op.xy,op.fixed,out);
  HXT1_CUDA(cudaGetLastError());
}

inline void s05b1_component_action(const S22D4DOperator&op,int component,
    const Real*scalarX,Real*scalarY,Real*embed,Real*fullAx){
  int nv=op.A.n;
  HXT1_CUDA(cudaMemset(embed,0,(std::size_t)3*nv*sizeof(Real)));
  s05b1_embed<<<(nv+255)/256,256>>>(nv,component,scalarX,embed);
  if(s05b2_fast_enabled())
    s05b2_component_action_fast(op,component,embed,fullAx);
  else
    op.apply(embed,fullAx);
  HXT1_CUDA(cudaMemcpy(scalarY,fullAx+(std::size_t)component*nv,
    (std::size_t)nv*sizeof(Real),cudaMemcpyDeviceToDevice));
  const unsigned char*mask=(component==2?op.fixed:op.xy);
  hxt5e::zero_masked_kernel<<<(nv+255)/256,256>>>(nv,mask,scalarY);
  HXT1_CUDA(cudaGetLastError());
}
struct S05B1ComponentResult{int its=0,cycles=0;double initial=0,final=0;bool ok=false;};
inline S05B1ComponentResult s05b1_component_fgmres(const S22D4DOperator&op,
    int component,const Real*rhs,Real*delta,
    Real*embed,Real*fullAx,
    int restart,int maxits,int precondSweeps,double omega,
    double rtol,double atol,hxt4a::Reducer<Real>&red){
  const int n=op.A.n;
  restart=std::min(restart,maxits);
  auto&W=hxt5e::get_workspace(n,restart);
  const unsigned char*mask=(component==2?op.fixed:op.xy);
  const Real*diag=(component==0?op.v0:component==1?op.v1:op.v2);
  S05B1ComponentResult R{};
  HXT1_CUDA(cudaMemset(delta,0,(std::size_t)n*sizeof(Real)));
  R.initial=red.norm(rhs);R.final=R.initial;
  const double target=std::max(atol,rtol*R.initial);
  if(!std::isfinite(R.initial))throw std::runtime_error("Stage5.B.1 nonfinite component rhs");
  if(R.initial<=target){R.ok=true;return R;}
  while(R.its<maxits && R.final>target){
    ++R.cycles;
    s05b1_component_action(op,component,delta,W.w,embed,fullAx);
    hxt5e::masked_residual_kernel<<<(n+255)/256,256>>>(n,mask,rhs,W.w,W.r);
    HXT1_CUDA(cudaGetLastError());
    const double beta=red.norm(W.r);
    if(!std::isfinite(beta))throw std::runtime_error("Stage5.B.1 nonfinite component residual");
    if(beta<=target){R.final=beta;break;}
    hxt5e::copy_scaled_kernel<<<(n+255)/256,256>>>(n,W.r,W.v(0),(Real)(1./beta));
    std::vector<double>H((std::size_t)(restart+1)*restart,0.),
      cs((std::size_t)restart,0.),sn((std::size_t)restart,0.),g((std::size_t)restart+1,0.);
    g[0]=beta;
    auto h=[&](int i,int j)->double&{return H[(std::size_t)i*restart+(std::size_t)j];};
    int kdim=0;bool breakdown=false;
    for(int j=0;j<restart && R.its<maxits;++j){
      // The preconditioner may be approximate; Krylov's operator is exact K_ii.
      hxt5e::right_fgs(op.A,op.C,diag,mask,W.v(j),W.z(j),precondSweeps,omega);
      s05b1_component_action(op,component,W.z(j),W.w,embed,fullAx);
      for(int i=0;i<=j;++i){
        h(i,j)=red.dot(W.w,W.v(i));
        hxt5e::axpy_kernel<<<(n+255)/256,256>>>(n,(Real)(-h(i,j)),W.v(i),W.w);
      }
      const double hn=red.norm(W.w);
      if(!std::isfinite(hn))throw std::runtime_error("Stage5.B.1 nonfinite Arnoldi norm");
      h(j+1,j)=hn;
      if(hn>1.e-30)hxt5e::copy_scaled_kernel<<<(n+255)/256,256>>>(n,W.w,W.v(j+1),(Real)(1./hn));
      else breakdown=true;
      for(int i=0;i<j;++i){
        double a=h(i,j),b0=h(i+1,j);
        h(i,j)=cs[(std::size_t)i]*a+sn[(std::size_t)i]*b0;
        h(i+1,j)=-sn[(std::size_t)i]*a+cs[(std::size_t)i]*b0;
      }
      const double a=h(j,j),b0=h(j+1,j),rho=std::hypot(a,b0);
      cs[(std::size_t)j]=rho>0?a/rho:1.;sn[(std::size_t)j]=rho>0?b0/rho:0.;
      h(j,j)=rho;h(j+1,j)=0.;
      const double gj=g[(std::size_t)j];
      g[(std::size_t)j]=cs[(std::size_t)j]*gj;
      g[(std::size_t)j+1]=-sn[(std::size_t)j]*gj;
      ++R.its;kdim=j+1;
      if(std::abs(g[(std::size_t)j+1])<=target||breakdown)break;
    }
    if(kdim<=0)break;
    std::vector<double>y((std::size_t)kdim,0.);
    for(int i=kdim-1;i>=0;--i){
      double q=g[(std::size_t)i];
      for(int j=i+1;j<kdim;++j)q-=h(i,j)*y[(std::size_t)j];
      if(!(std::abs(h(i,i))>1.e-300))throw std::runtime_error("Stage5.B.1 singular Hessenberg");
      y[(std::size_t)i]=q/h(i,i);
    }
    for(int j=0;j<kdim;++j)
      hxt5e::axpy_kernel<<<(n+255)/256,256>>>(n,(Real)y[(std::size_t)j],W.z(j),delta);
    hxt5e::zero_masked_kernel<<<(n+255)/256,256>>>(n,mask,delta);
    s05b1_component_action(op,component,delta,W.w,embed,fullAx);
    hxt5e::masked_residual_kernel<<<(n+255)/256,256>>>(n,mask,rhs,W.w,W.r);
    R.final=red.norm(W.r);
    if(!std::isfinite(R.final))throw std::runtime_error("Stage5.B.1 nonfinite component true residual");
    if(breakdown)break;
  }
  R.ok=R.final<=target;
  return R;
}
inline S22D4DResult s05b1_segregated_fgmres(const S22D4DOperator&op,
    Real*x,const Real*b,Real*embed,Real*ax,Real*residual,
    Real*backup,Real*rhs,Real*delta,
    int blockMax,int componentMax,int restart,int precondSweeps,double omega,
    double rtol,double atol,hxt4a::Reducer<Real>&redFull,
    hxt4a::Reducer<Real>&redScalar,int outer){
  const int nv=op.A.n;
  S22D4DResult R{};
  s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,op.xy,op.fixed,x);
  if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(
    op.nfaces,op.faces,op.tvel,x,x+nv,x+2*(std::size_t)nv);
  HXT1_CUDA(cudaMemcpy(backup,x,(std::size_t)3*nv*sizeof(Real),cudaMemcpyDeviceToDevice));
  R.initial=s05b1_full_residual(op,b,x,ax,residual,redFull,true);
  R.final=R.initial;
  const double target=std::max(atol,rtol*R.initial);
  if(!std::isfinite(R.initial))throw std::runtime_error("Stage5.B.1 nonfinite initial full momentum residual");
  const unsigned char*rhsMasks[3]={op.xy,op.xy,op.fixed};
  for(int iblock=0;iblock<blockMax && R.final>target;++iblock){
    const double before=R.final;
    for(int ic=0;ic<3;++ic){
      (void)s05b1_full_residual(op,b,x,ax,residual,redFull,false);
      HXT1_CUDA(cudaMemcpy(rhs,residual+(std::size_t)ic*nv,
        (std::size_t)nv*sizeof(Real),cudaMemcpyDeviceToDevice));
      hxt5e::zero_masked_kernel<<<(nv+255)/256,256>>>(nv,rhsMasks[ic],rhs);
      S05B1ComponentResult q=s05b1_component_fgmres(op,ic,rhs,delta,
        embed,ax,restart,componentMax,precondSweeps,omega,rtol,atol,redScalar);
      // Always validate component true residual, but block success is judged
      // ONLY with the true full three-component projected operator residual.
      s05b1_add<<<(nv+255)/256,256>>>(nv,ic,delta,x);
      s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,op.xy,op.fixed,x);
      if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(
        op.nfaces,op.faces,op.tvel,x,x+nv,x+2*(std::size_t)nv);
      R.its+=q.its;
      std::printf("STAGE27_S05B1_COMPONENT it=%d block=%d comp=%c its=%d cycles=%d "
        "trueRel=%.8e converged=%d operator=Eit_TtAT_Ei status=%s\n",
        outer,iblock+1,"xyz"[ic],q.its,q.cycles,
        q.final/std::max(q.initial,1.e-300),(int)q.ok,q.ok?"PASS":"WARN");
    }
    R.cycles++;
    R.final=s05b1_full_residual(op,b,x,ax,residual,redFull,true);
    std::printf("STAGE27_S05B1_BLOCK it=%d pass=%d initial=%.12e current=%.12e "
      "trueRel=%.12e ratioThisPass=%.6e operator=Tt_A_T status=%s\n",
      outer,R.cycles,R.initial,R.final,R.final/std::max(R.initial,1.e-300),
      R.final/std::max(before,1.e-300),
      std::isfinite(R.final)?"PASS":"WARN");
    if(!std::isfinite(R.final)||R.final>before*2.)break;
  }
  R.rel=R.final/std::max(R.initial,1.e-300);
  R.ok=std::isfinite(R.final)&&R.final<=target;
  if(!R.ok){
    // Roll back *all* inexact block passes before any robust fallback.
    HXT1_CUDA(cudaMemcpy(x,backup,(std::size_t)3*nv*sizeof(Real),cudaMemcpyDeviceToDevice));
    std::printf("STAGE27_S05B1_RESTORE it=%d reason=BLOCK_NOT_CONVERGED "
      "initial=%.12e attempted=%.12e rtol=%.3e status=PASS\n",
      outer,R.initial,R.final,rtol);
  }
  std::printf("STAGE27_S05B1_RESULT it=%d its=%d blockPasses=%d "
    "trueRel=%.12e accepted=%d status=%s\n",outer,R.its,R.cycles,
    R.rel,(int)R.ok,R.ok?"PASS":"WARN");
  return R;
}

// Stage27/5.B.3: scalar low-memory BiCGSTAB and fixed FGS-preconditioned Richardson.
// Both use the EXACT Stage5.B.2 diagonal projected component action Kii=Eit Tt A T Ei.
// Original segregated/coupled FGMRES are untouched. No 3-component Krylov vectors.
// Fixed mode performs NO scalar reductions inside sweeps; full-vector residual
// gates can be disabled for validated deliberately-inexact operation.
struct S05B3Workspace {
  int n;
  // nine scalar buffers: r, rhat, p, v, s, t, z, y, Ax.
  Real *base=nullptr; Real *r,*rhat,*p,*v,*s,*t,*z,*y,*ax;
  explicit S05B3Workspace(int num):n(num){
    HXT1_CUDA(cudaMalloc((void**)&base,(std::size_t)9*n*sizeof(Real)));
    r=base;rhat=r+n;p=rhat+n;v=p+n;s=v+n;t=s+n;z=t+n;y=z+n;ax=y+n;
    std::printf("STAGE27_S05B3_WORKSPACE scalarDofs=%d bytes=%zu MiB=%.3f "
      "buffers=9 allocatedOnDemand=YES status=PASS\n",n,
      (std::size_t)9*n*sizeof(Real),9.*n*sizeof(Real)/1048576.);
  }
  ~S05B3Workspace(){if(base)cudaFree(base);}
  S05B3Workspace(const S05B3Workspace&)=delete;
  S05B3Workspace&operator=(const S05B3Workspace&)=delete;
};
inline S05B3Workspace& s05b3_workspace(int n){
  static std::vector<std::unique_ptr<S05B3Workspace>>cache;
  for(auto&v:cache)if(v->n==n)return *v;
  cache.emplace_back(std::make_unique<S05B3Workspace>(n));
  return *cache.back();
}
__global__ void s05b3_bicg_p(int n,Real*r,Real*p,const Real*v,Real beta,Real omega){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)p[i]=r[i]+beta*(p[i]-omega*v[i]);
}
__global__ void s05b3_bicg_s(int n,const Real*r,const Real*v,Real alpha,Real*s){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)s[i]=r[i]-alpha*v[i];
}
__global__ void s05b3_bicg_update(int n,Real*x,const Real*y,const Real*z,
    const Real*s,const Real*t,Real alpha,Real omega,Real*r){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){x[i]+=alpha*y[i]+omega*z[i];r[i]=s[i]-omega*t[i];}
}
__global__ void s05b3_relax(int n,Real*x,const Real*z,Real damping){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)x[i]+=damping*z[i];
}
inline S05B1ComponentResult s05b3_component_bicgstab(
    const S22D4DOperator&op,int comp,const Real*rhs,Real*delta,
    Real*embed,Real*fullAx,int maxit,int precondSweeps,double fgsOmega,
    double rtol,double atol,hxt4a::Reducer<Real>&red){
  const int n=op.A.n;auto&W=s05b3_workspace(n);
  const unsigned char*mask=(comp==2?op.fixed:op.xy);
  const Real*diag=(comp==0?op.v0:comp==1?op.v1:op.v2);
  S05B1ComponentResult R{};
  HXT1_CUDA(cudaMemset(delta,0,(std::size_t)n*sizeof(Real)));
  HXT1_CUDA(cudaMemcpy(W.r,rhs,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
  hxt5e::zero_masked_kernel<<<(n+255)/256,256>>>(n,mask,W.r);
  HXT1_CUDA(cudaMemcpy(W.rhat,W.r,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
  R.initial=red.norm(W.r);R.final=R.initial;
  const double target=std::max(atol,rtol*R.initial);
  if(!std::isfinite(R.initial))return R;
  if(R.initial<=target){R.ok=true;return R;}
  double rhoOld=1.,alpha=1.,omega=1.;
  for(int it=0;it<maxit;++it){
    const double rho=red.dot(W.rhat,W.r);
    if(!std::isfinite(rho)||std::abs(rho)<1.e-300)break;
    if(it==0){HXT1_CUDA(cudaMemcpy(W.p,W.r,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));}
    else{
      if(!std::isfinite(omega)||std::abs(omega)<1.e-300)break;
      const Real beta=(Real)((rho/rhoOld)*(alpha/omega));
      s05b3_bicg_p<<<(n+255)/256,256>>>(n,W.r,W.p,W.v,beta,(Real)omega);
    }
    hxt5e::right_fgs(op.A,op.C,diag,mask,W.p,W.y,precondSweeps,fgsOmega);
    s05b1_component_action(op,comp,W.y,W.v,embed,fullAx);
    const double denom=red.dot(W.rhat,W.v);
    if(!std::isfinite(denom)||std::abs(denom)<1.e-300)break;
    alpha=rho/denom;
    if(!std::isfinite(alpha))break;
    s05b3_bicg_s<<<(n+255)/256,256>>>(n,W.r,W.v,(Real)alpha,W.s);
    const double sNorm=red.norm(W.s);
    if(!std::isfinite(sNorm))break;
    if(sNorm<=target){
      hxt5e::axpy_kernel<<<(n+255)/256,256>>>(n,(Real)alpha,W.y,delta);
      ++R.its;R.final=sNorm;break;
    }
    hxt5e::right_fgs(op.A,op.C,diag,mask,W.s,W.z,precondSweeps,fgsOmega);
    s05b1_component_action(op,comp,W.z,W.t,embed,fullAx);
    const double tt=red.dot(W.t,W.t);
    if(!std::isfinite(tt)||tt<1.e-300)break;
    omega=red.dot(W.t,W.s)/tt;
    if(!std::isfinite(omega)||std::abs(omega)<1.e-300)break;
    s05b3_bicg_update<<<(n+255)/256,256>>>(n,delta,W.y,W.z,W.s,W.t,
      (Real)alpha,(Real)omega,W.r);
    R.its++;
    R.final=red.norm(W.r);
    if(!std::isfinite(R.final)||R.final<=target)break;
    rhoOld=rho;
  }
  // True Kii residual is the acceptance criterion, not the BiCG recurrence norm.
  hxt5e::zero_masked_kernel<<<(n+255)/256,256>>>(n,mask,delta);
  s05b1_component_action(op,comp,delta,W.ax,embed,fullAx);
  hxt5e::masked_residual_kernel<<<(n+255)/256,256>>>(n,mask,rhs,W.ax,W.r);
  R.final=red.norm(W.r);
  R.cycles=R.its?1:0;R.ok=std::isfinite(R.final)&&R.final<=target;
  return R;
}
inline S05B1ComponentResult s05b3_component_fgs_richardson(
    const S22D4DOperator&op,int comp,const Real*rhs,Real*delta,
    Real*embed,Real*fullAx,int sweeps,int fgsSweeps,double fgsOmega,
    double damping,bool audit,hxt4a::Reducer<Real>&red){
  const int n=op.A.n;auto&W=s05b3_workspace(n);
  const unsigned char*mask=(comp==2?op.fixed:op.xy);
  const Real*diag=(comp==0?op.v0:comp==1?op.v1:op.v2);
  S05B1ComponentResult R{};
  HXT1_CUDA(cudaMemset(delta,0,(std::size_t)n*sizeof(Real)));
  // Each sweep: z = FGS(Aii_unprojected)^-1 (rhs - Kii*delta);
  // delta <- delta + damping*z. This is a projected-operator stationary
  // iteration, NOT literal GS on an assembled Kii CSR matrix.
  for(int sweep=0;sweep<sweeps;++sweep){
    s05b1_component_action(op,comp,delta,W.ax,embed,fullAx);
    hxt5e::masked_residual_kernel<<<(n+255)/256,256>>>(n,mask,rhs,W.ax,W.r);
    hxt5e::right_fgs(op.A,op.C,diag,mask,W.r,W.z,fgsSweeps,fgsOmega);
    s05b3_relax<<<(n+255)/256,256>>>(n,delta,W.z,(Real)damping);
  }
  hxt5e::zero_masked_kernel<<<(n+255)/256,256>>>(n,mask,delta);
  R.its=sweeps;R.cycles=1;
  if(audit){
    s05b1_component_action(op,comp,delta,W.ax,embed,fullAx);
    hxt5e::masked_residual_kernel<<<(n+255)/256,256>>>(n,mask,rhs,W.ax,W.r);
    R.initial=red.norm(rhs);R.final=red.norm(W.r);
    R.ok=std::isfinite(R.final)&&R.final<=R.initial;
  }else{R.initial=R.final=std::numeric_limits<double>::quiet_NaN();R.ok=true;}
  return R;
}
inline S22D4DResult s05b3_segregated(const S22D4DOperator&op,
    Real*x,const Real*b,Real*embed,Real*ax,Real*residual,
    Real*backup,Real*rhs,Real*delta,
    int blockPasses,int componentIts,int innerFgs,double omega,double damping,
    bool bicgstab,bool fullAudit,bool requireTolerance,double rtol,double atol,
    hxt4a::Reducer<Real>&redFull,hxt4a::Reducer<Real>&redScalar,int outer){
  const int nv=op.A.n;S22D4DResult R{};
  s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,op.xy,op.fixed,x);
  if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(
    op.nfaces,op.faces,op.tvel,x,x+nv,x+2*(std::size_t)nv);
  HXT1_CUDA(cudaMemcpy(backup,x,(std::size_t)3*nv*sizeof(Real),cudaMemcpyDeviceToDevice));
  // One baseline full-system norm at the start; optional checks at block boundaries.
  R.initial=s05b1_full_residual(op,b,x,ax,residual,redFull,true);
  R.final=R.initial;
  const double target=std::max(atol,rtol*R.initial);
  if(!std::isfinite(R.initial))throw std::runtime_error("Stage5.B.3 nonfinite initial full momentum residual");
  const unsigned char*mask[3]={op.xy,op.xy,op.fixed};
  int componentFailures=0;
  for(int ib=0;ib<blockPasses;++ib){
    if(requireTolerance && fullAudit && R.final<=target)break;
    for(int ic=0;ic<3;++ic){
      (void)s05b1_full_residual(op,b,x,ax,residual,redFull,false);
      HXT1_CUDA(cudaMemcpy(rhs,residual+(std::size_t)ic*nv,
        (std::size_t)nv*sizeof(Real),cudaMemcpyDeviceToDevice));
      hxt5e::zero_masked_kernel<<<(nv+255)/256,256>>>(nv,mask[ic],rhs);
      const auto q=bicgstab?
        s05b3_component_bicgstab(op,ic,rhs,delta,embed,ax,componentIts,innerFgs,omega,rtol,atol,redScalar):
        s05b3_component_fgs_richardson(op,ic,rhs,delta,embed,ax,componentIts,innerFgs,omega,damping,fullAudit,redScalar);
      if(!q.ok)++componentFailures;
      s05b1_add<<<(nv+255)/256,256>>>(nv,ic,delta,x);
      s22d4d_zero_fixed<<<(3*nv+255)/256,256>>>(nv,op.xy,op.fixed,x);
      if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(
        op.nfaces,op.faces,op.tvel,x,x+nv,x+2*(std::size_t)nv);
      R.its+=q.its;
      std::printf("STAGE27_S05B3_COMPONENT it=%d pass=%d comp=%c solver=%s "
        "its=%d innerTrueRel=%.9e audited=%d status=%s\n",outer,ib+1,"xyz"[ic],
        bicgstab?"bicgstab":"fgs-richardson",q.its,
        q.final/std::max(q.initial,1.e-300),(int)(bicgstab||fullAudit),q.ok?"PASS":"WARN");
    }
    R.cycles++;
    if(fullAudit){
      const double previous=R.final;
      R.final=s05b1_full_residual(op,b,x,ax,residual,redFull,true);
      std::printf("STAGE27_S05B3_BLOCK it=%d pass=%d initial=%.12e current=%.12e "
        "trueRel=%.12e ratioThisPass=%.6e fullAudit=YES status=%s\n",
        outer,ib+1,R.initial,R.final,R.final/std::max(R.initial,1.e-300),
        R.final/std::max(previous,1.e-300),std::isfinite(R.final)?"PASS":"WARN");
      if(!std::isfinite(R.final)||R.final>previous*2.)break;
    }
  }
  // Always do one final guard/check for the benchmark; no reductions during FGS sweeps.
  R.final=s05b1_full_residual(op,b,x,ax,residual,redFull,true);
  R.rel=R.final/std::max(R.initial,1.e-300);
  const bool finite=std::isfinite(R.final);
  const bool safe=finite && R.final<=R.initial*1.25;
  R.ok= safe && (!requireTolerance || R.final<=target);
  if(!R.ok){
    HXT1_CUDA(cudaMemcpy(x,backup,(std::size_t)3*nv*sizeof(Real),cudaMemcpyDeviceToDevice));
    std::printf("STAGE27_S05B3_RESTORE it=%d reason=%s fullRel=%.12e "
      "requireTolerance=%d status=PASS\n",outer,finite?"FULL_OPERATOR_GATE":"NONFINITE",
      R.rel,(int)requireTolerance);
  }
  std::printf("STAGE27_S05B3_RESULT it=%d solver=%s its=%d blockPasses=%d "
    "trueRel=%.12e componentWarnings=%d accepted=%d fullAudit=%d "
    "inexact=%d status=%s\n",outer,bicgstab?"bicgstab":"fgs-richardson",
    R.its,R.cycles,R.rel,componentFailures,(int)R.ok,(int)fullAudit,
    (int)!requireTolerance,R.ok?"PASS":"WARN");
  return R;
}


// M3: TRUE BF3-PROJECTED COUPLED BLOCK CSR, numerically assembled every
// SIMPLE step from the current 3 scalar CSR values and F1B frozen face cache.
// Exact algebra: M=diag(Ax,Ay,Az)+W; K=T^T M T.
// Three-component 3x3 CSR blocks share original scalar CSR sparsity.
// No element quadrature and no projected full-operator action IN GS SWEEPS.
// The right and left transforms use immutable sources and atomic scatter to
// avoid race-dependent feedback between different constrained wall bubbles.
// This is a FIRST PERFORMANCE EXPERIMENT: 18*nnz*8 bytes temporary VRAM.
__global__ void m3_init_blocks(std::size_t nnz,
    const Real*ax,const Real*ay,const Real*az,Real*m){
  std::size_t k=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=nnz)return;
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)
    m[((std::size_t)a*3+b)*nnz+k]=(a==b?(a==0?ax[k]:(a==1?ay[k]:az[k])):Real(0));
}
__global__ void m3_add_f1b(int nf,const hxt5b::TetFacePlan*f,
    const std::int32_t*tetSlot,const double*faceC,std::size_t nnz,Real*m){
  int fi=(int)blockIdx.x,k=(int)threadIdx.x;
  if(fi>=nf||k>=64)return;
  const auto&face=f[fi];
  const int slot=tetSlot[(std::size_t)face.cell*64u+(std::size_t)k];
  const double c=faceC[(std::size_t)fi*64u+(std::size_t)k];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    if(i!=j)atomicAdd(m+((std::size_t)i*3+j)*nnz+slot,
       (Real)(face.n[i]*face.n[j]*c));
  }
}
__device__ __forceinline__ int m3_lookup(const std::int64_t*row,
     const std::int32_t*col,int i,int j){
  std::int64_t a=row[i],z=row[i+1];
  while(a<z){std::int64_t mid=(a+z)/2;if(col[mid]<j)a=mid+1;else z=mid;}
  return (a<row[i+1] && col[a]==j)?(int)a:-1;
}
__global__ void m3_right_transform(int nf,const hxt5b::TetFacePlan*faces,
     const nodals_hxt1::TetVel*tv,const std::int64_t*row,
     const std::int32_t*col,std::size_t nnz,const Real*src,Real*dst,
     unsigned long long*bad){
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=faces[fi];const auto v=tv[f.cell];const int bubble=v.g[4+(int)f.lf];
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const int faceVert[3]={
    v.g[hxt2::c_tet_face[(int)f.lf][0]],
    v.g[hxt2::c_tet_face[(int)f.lf][1]],
    v.g[hxt2::c_tet_face[(int)f.lf][2]]};
  for(std::int64_t k=row[bubble]+threadIdx.x;k<row[bubble+1];k+=blockDim.x){
    const int i=col[k];const int ib=m3_lookup(row,col,i,bubble);
    if(ib<0){atomicAdd(bad,1ULL);continue;}
    int iv[3];for(int a=0;a<3;++a)iv[a]=m3_lookup(row,col,i,faceVert[a]);
    for(int r=0;r<3;++r){
      double mn=0.;for(int c=0;c<3;++c)
        mn+=(double)src[((std::size_t)r*3+c)*nnz+ib]*n[c];
      for(int c=0;c<3;++c){
        const double delta=-mn*n[c];
        atomicAdd(dst+((std::size_t)r*3+c)*nnz+ib,(Real)delta);
        for(int a=0;a<3;++a){if(iv[a]<0){atomicAdd(bad,1ULL);continue;}
          atomicAdd(dst+((std::size_t)r*3+c)*nnz+iv[a],(Real)((20./27.)*delta));}
      }
    }
  }
}
__global__ void m3_left_transform(int nf,const hxt5b::TetFacePlan*faces,
     const nodals_hxt1::TetVel*tv,const std::int64_t*row,
     const std::int32_t*col,std::size_t nnz,const Real*src,Real*dst,
     unsigned long long*bad){
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=faces[fi];const auto v=tv[f.cell];const int bubble=v.g[4+(int)f.lf];
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const int faceVert[3]={
    v.g[hxt2::c_tet_face[(int)f.lf][0]],
    v.g[hxt2::c_tet_face[(int)f.lf][1]],
    v.g[hxt2::c_tet_face[(int)f.lf][2]]};
  for(std::int64_t k=row[bubble]+threadIdx.x;k<row[bubble+1];k+=blockDim.x){
    const int j=col[k];
    int vj[3];for(int a=0;a<3;++a)vj[a]=m3_lookup(row,col,faceVert[a],j);
    for(int c=0;c<3;++c){
      double nm=0.;for(int r=0;r<3;++r)
        nm+=n[r]*(double)src[((std::size_t)r*3+c)*nnz+k];
      for(int r=0;r<3;++r){
        const double delta=-n[r]*nm;
        atomicAdd(dst+((std::size_t)r*3+c)*nnz+k,(Real)delta);
        for(int a=0;a<3;++a){if(vj[a]<0){atomicAdd(bad,1ULL);continue;}
          atomicAdd(dst+((std::size_t)r*3+c)*nnz+vj[a],(Real)((20./27.)*delta));}
      }
    }
  }
}
__global__ void m3_spmv(int nv,std::size_t nnz,
    const std::int64_t*row,const std::int32_t*col,
    const Real*k,const Real*x,Real*y,
    const unsigned char*maskXY,const unsigned char*maskZ){
  int i=(int)blockIdx.x*blockDim.x+threadIdx.x;if(i>=nv)return;
  double sum[3]={0,0,0};
  for(auto p=row[i];p<row[i+1];++p){
    const int j=col[p];const double x0=x[j],x1=x[(std::size_t)nv+j],x2=x[2*(std::size_t)nv+j];
    for(int c=0;c<3;++c)sum[c]+=(double)k[((std::size_t)c*3+0)*nnz+p]*x0+
      (double)k[((std::size_t)c*3+1)*nnz+p]*x1+
      (double)k[((std::size_t)c*3+2)*nnz+p]*x2;
  }
  for(int c=0;c<3;++c)y[(std::size_t)c*nv+i]=
     (c==2?maskZ[i]:maskXY[i])?Real(0):(Real)sum[c];
}
__global__ void m3_sub(int n,const Real*a,const Real*b,Real*out){
  int i=(int)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a[i]-b[i];
}
// A 3x3 local block solve. BF3 wall bubbles carry only two tangential
// unknowns, so give their null normal a positive diagonal and project the
// correction back into the tangential plane; exact tangential block remains.
__device__ __forceinline__ bool m3_solve3(double a[3][3],double rhs[3],double sol[3]){
  for(int q=0;q<3;++q){
    int pivot=q;for(int j=q+1;j<3;++j)if(fabs(a[j][q])>fabs(a[pivot][q]))pivot=j;
    const double v=fabs(a[pivot][q]);if(!(v>1.e-27)||!isfinite(v))return false;
    if(pivot!=q){for(int j=q;j<3;++j){double t=a[q][j];a[q][j]=a[pivot][j];a[pivot][j]=t;}
      double t=rhs[q];rhs[q]=rhs[pivot];rhs[pivot]=t;}
    for(int i=q+1;i<3;++i){const double f=a[i][q]/a[q][q];
      for(int j=q;j<3;++j)a[i][j]-=f*a[q][j];rhs[i]-=f*rhs[q];}
  }
  for(int i=2;i>=0;--i){double v=rhs[i];for(int j=i+1;j<3;++j)v-=a[i][j]*sol[j];sol[i]=v/a[i][i];}
  return isfinite(sol[0])&&isfinite(sol[1])&&isfinite(sol[2]);
}
__global__ void m3_block_colour(int start,int count,const std::int32_t*rows,
    const std::int64_t*row,const std::int32_t*col,const std::int32_t*diag,
    const Real*k,std::size_t nnz,int nv,const Real*b,Real*x,
    const unsigned char*maskXY,const unsigned char*maskZ,
    const int*faceByBubble,const hxt5b::TetFacePlan*faces,
    double omega,unsigned long long*bad){
  int q=(int)blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;
  int i=rows[start+q];
  const bool fixed[3]={maskXY[i]!=0,maskXY[i]!=0,maskZ[i]!=0};
  if(fixed[0]&&fixed[1]&&fixed[2])return;
  double rhs[3]={double(b[i]),double(b[(std::size_t)nv+i]),double(b[2*(std::size_t)nv+i])};
  for(auto p=row[i];p<row[i+1];++p){
    const int j=col[p];const double xx[3]={double(x[j]),double(x[(std::size_t)nv+j]),double(x[2*(std::size_t)nv+j])};
    for(int a=0;a<3;++a)for(int c=0;c<3;++c)
      rhs[a]-=(double)k[((std::size_t)a*3+c)*nnz+p]*xx[c];
  }
  double a[3][3];const int di=diag[i];
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[r][c]=(double)k[((std::size_t)r*3+c)*nnz+di];
  const int fi=faceByBubble[i];
  double n[3]={0,0,0};
  if(fi>=0){
    const auto f=faces[fi];for(int c=0;c<3;++c)n[c]=f.n[c];
    const double scale=fmax(1.e-20,fmax(fabs(a[0][0]),fmax(fabs(a[1][1]),fabs(a[2][2]))));
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[r][c]+=scale*n[r]*n[c];
    // The reduced parameter RHS is already tangent up to FP round-off.
    const double rn=n[0]*rhs[0]+n[1]*rhs[1]+n[2]*rhs[2];
    for(int c=0;c<3;++c)rhs[c]-=rn*n[c];
  }
  for(int c=0;c<3;++c)if(fixed[c]){
    for(int j=0;j<3;++j){a[c][j]=0;a[j][c]=0;}a[c][c]=1;rhs[c]=0;
  }
  double sol[3]={0,0,0};
  if(!m3_solve3(a,rhs,sol)){
    atomicAdd(bad,1ULL);return;
  }
  if(fi>=0){const double sn=n[0]*sol[0]+n[1]*sol[1]+n[2]*sol[2];
    for(int c=0;c<3;++c)sol[c]-=sn*n[c];}
  for(int c=0;c<3;++c)if(!fixed[c])x[(std::size_t)c*nv+i]+=(Real)(omega*sol[c]);
}
// NODALS_EMAC_FULL_M3_GERSHGORIN_RELAX_V1
// Optional pseudo-transient/linearization stabilization of the COMPLETE reduced
// nine-block M3 matrix. D is added to K; the SAME D*x_old is added to the RHS.
// Thus K*x=b is unchanged at a converged outer fixed point.
// Unlike scalar-only row L1, the correction includes every cross-component row.
__global__ void m3_emac_full_row_l1(int n,std::size_t nnz,
    const std::int64_t*row,const std::int32_t*diag,
    Real*k,const Real*x,const Real*b,
    const unsigned char*xy,const unsigned char*z,
    double factor,Real*shift,Real*rhs,Real*originalDiag,unsigned long long*bad){
  const int i=(int)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const auto di=(std::size_t)diag[i];
  for(int a=0;a<3;++a){
    const std::size_t ix=(std::size_t)a*(std::size_t)n+i;
    const bool fixed=(a==2?z[i]:xy[i])!=0;
    const std::size_t kd=((std::size_t)a*3+a)*nnz+di;
    originalDiag[ix]=k[kd];
    double delta=0.0;
    if(!fixed){
      double off=0.0;
      for(auto p=row[i];p<row[i+1];++p)
        for(int c=0;c<3;++c)
          if(c!=a || (std::size_t)p!=di)
            off+=fabs((double)k[((std::size_t)a*3+c)*nnz+(std::size_t)p]);
      const double diagonal=(double)k[kd];
      const double required=factor*off + 1.e-12;
      delta=fmax(0.0,required-diagonal);
      if(!isfinite(delta)||!isfinite(diagonal)||!isfinite(off)){
        atomicAdd(bad,1ULL);delta=0.0;
      }
      k[kd]+=(Real)delta;
    }
    shift[ix]=(Real)delta;
    rhs[ix]=b[ix]+(Real)delta*x[ix];
  }
}
__global__ void m3_emac_full_row_l1_undo(int n,std::size_t nnz,
    const std::int32_t*diag,Real*k,const Real*originalDiag){
  const int i=(int)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const std::size_t di=(std::size_t)diag[i];
  for(int a=0;a<3;++a){
    const std::size_t ix=(std::size_t)a*(std::size_t)n+i;
    k[((std::size_t)a*3+a)*nnz+di]=originalDiag[ix];
  }
}

inline void m3_build(const S22D4DOperator&op,Real*k,Real*tmp,
    unsigned long long*bad){
  const std::size_t nnz=op.A.nnz,bytes=9ull*nnz*sizeof(Real);
  HXT1_CUDA(cudaMemset(bad,0,sizeof(unsigned long long)));
  m3_init_blocks<<<(nnz+255)/256,256>>>(nnz,op.v0,op.v1,op.v2,k);
  m3_add_f1b<<<op.nfaces,64>>>(op.nfaces,op.faces,
    op.A.tetSlot.p,op.f1bFaceCoupling,nnz,k);
  if(op.bazbViscosityMode)
    bazb_m3_add<<<op.nfaces,64>>>(op.faces,op.nfaces,op.quads,
       op.tgeom,op.A.tetSlot.p,op.coeff,op.bazbMolecularNu,
       op.bazbViscosityMode,nnz,k,bad);
  if(op.genrobinViscosityMode)
    genrobin_m3_add<<<op.nfaces,64>>>(
       op.faces,op.nfaces,op.quads,op.tgeom,op.A.tetSlot.p,
       op.coeff,op.genrobinMolecularNu,op.genrobinViscosityMode,
       op.genrobinC,nnz,k,bad);
  // NODALS_EMAC_PICARD_V1: add EMAC gradient-of-energy and divergence extra blocks.
  // Existing advective A.conv supplies the first term.
  if(op.emacPicard)
    emac_picard_tet_add_m3<<<op.emacNTet,64>>>(
      op.tvel,op.tgeom,op.A.tetSlot.p,
      op.emacU0,op.emacU1,op.emacU2,op.emacNTet,nnz,k);
  // Add 3x3 cross/divergence tensor to FULL M3 CSR before BF3 T/Tt.
  // This is an implicit matrix term, NOT a lagged RHS correction.
  if(op.implicitDev)
    implicit_dev_tet_add_m3<<<op.devNTet,64>>>(
      op.devPoints,op.devTetConn,op.devTetVel,op.devTetGeom,
      op.A.tetSlot.p,op.devU0,op.devU1,op.devU2,op.devK,op.devOmega,
      op.devNTet,op.devRadius,op.devNu,op.devKFloor,op.devOmegaFloor,
      nnz,k);
  HXT1_CUDA(cudaMemcpy(tmp,k,bytes,cudaMemcpyDeviceToDevice));
  m3_right_transform<<<op.nfaces,128>>>(op.nfaces,op.faces,op.tvel,
    op.A.row.p,op.A.col.p,nnz,k,tmp,bad);
  HXT1_CUDA(cudaMemcpy(k,tmp,bytes,cudaMemcpyDeviceToDevice));
  m3_left_transform<<<op.nfaces,128>>>(op.nfaces,op.faces,op.tvel,
    op.A.row.p,op.A.col.p,nnz,tmp,k,bad);
  HXT1_CUDA(cudaGetLastError());
  HXT1_CUDA(cudaDeviceSynchronize());
  unsigned long long errors=0;
  HXT1_CUDA(cudaMemcpy(&errors,bad,sizeof(errors),cudaMemcpyDeviceToHost));
  if(errors)throw std::runtime_error("M3 CSR closure/assembly error count="+std::to_string(errors));
}
inline S22D4DResult m3_fixed_coupled_csr(const S22D4DOperator&op,
    Real*x,const Real*b,Real*k,Real*tmp,int sweeps,double omega,bool symmetric,
    const int*faceByBubble,unsigned long long*bad,int outer,
    hxt4a::Reducer<Real>*csrDiagnosticReducer,Real*csrDiagnosticAx,
    Real*csrDiagnosticResidual){
  // NODALS_EMAC_M3_TRUE_CSR_RESIDUAL_DIAG_V1: pure diagnostic; assembly and sweeps are unchanged.
  if(!op.bf3FluxMode||!op.f1bFaceCoupling)
    throw std::runtime_error("M3 requires BF3 and F1B cached wall");
  const int n=op.A.n;S22D4DResult R{};
  // The only constraint projection: at entry; every local bubble update remains tangent.
  s22d4d_zero_fixed<<<(3*n+255)/256,256>>>(n,op.xy,op.fixed,x);
  s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
      x,x+n,x+2*(std::size_t)n);
  const auto st=std::chrono::steady_clock::now();
  m3_build(op,k,tmp,bad);
  if(op.emacPicard && (outer==1 || outer%10==0))
    std::printf("NODALS_EMAC_M3_ACTIVE it=%d carrier=PICARD_U_STAR "
      "tensor=FULL_9_BLOCK_CSR bf3=TTMT pressure=SHIFTED status=PASS\n",outer);
  const double buildMs=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-st).count();
  // NODALS_EMAC_M3_TRUE_CSR_RESIDUAL_DIAG_V1: measure the ACTUAL frozen 9-block CSR K, not a rebuilt
  // matrix-free operator whose carrier might have changed later.
  // NODALS_EMAC_FULL_M3_GERSHGORIN_RELAX_V1: disabled unless explicitly requested via environment.
  // 9*nnz scratch is no longer needed after m3_build; use its first 9*n
  // entries for shifts, shifted RHS, and EXACT original diagonals.
  const char*envRelax=std::getenv("NODALS_EMAC_M3_ROW_L1_FACTOR");
  const double rowL1Factor=(op.emacPicard && envRelax && *envRelax)
    ? std::strtod(envRelax,nullptr) : 0.0;
  if(!std::isfinite(rowL1Factor)||rowL1Factor<0.0||rowL1Factor>10.0)
    throw std::runtime_error("NODALS_EMAC_M3_ROW_L1_FACTOR must be in [0,10]");
  const bool emacRowL1=(op.emacPicard && rowL1Factor>0.0);
  if(emacRowL1 && 9ull*op.A.nnz < 9ull*(std::size_t)n)
    throw std::runtime_error("M3 temporary scratch insufficient for row L1");
  Real*rowL1Shift=tmp;
  Real*rowL1Rhs=tmp+3*(std::size_t)n;
  Real*rowL1OriginalDiag=tmp+6*(std::size_t)n;
  const bool csrDiagnose=(csrDiagnosticReducer && csrDiagnosticAx &&
    csrDiagnosticResidual && outer<=2);
  auto frozenCsrResidual=[&](const Real*rhs){
    m3_spmv<<<(n+255)/256,256>>>(n,op.A.nnz,op.A.row.p,op.A.col.p,
      k,x,csrDiagnosticAx,op.xy,op.fixed);
    s22d4d_residual<<<(3*n+255)/256,256>>>(n,rhs,csrDiagnosticAx,
      op.xy,op.fixed,csrDiagnosticResidual);
    HXT1_CUDA(cudaGetLastError());
    return csrDiagnosticReducer->norm(csrDiagnosticResidual);
  };
  const double csrInitial=csrDiagnose?frozenCsrResidual(b):0.0;
  const Real*solveRhs=b;
  if(emacRowL1){
    HXT1_CUDA(cudaMemset(bad,0,sizeof(unsigned long long)));
    m3_emac_full_row_l1<<<(n+255)/256,256>>>(n,op.A.nnz,
      op.A.row.p,op.A.diagPos.p,k,x,b,op.xy,op.fixed,
      rowL1Factor,rowL1Shift,rowL1Rhs,rowL1OriginalDiag,bad);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long errors=0;
    HXT1_CUDA(cudaMemcpy(&errors,bad,sizeof(errors),cudaMemcpyDeviceToHost));
    if(errors)throw std::runtime_error("EMAC full row L1 invalid shifts count="+std::to_string(errors));
    solveRhs=rowL1Rhs;
    if(outer<=2)std::printf("NODALS_EMAC_FULL_M3_ROW_L1 it=%d factor=%.6f "
      "matrix=REDUCED_9BLOCK_CSR rhs=ADD_D_TIMES_OLD_U mode=OPT_IN status=PASS\n",
      outer,rowL1Factor);
  }
  HXT1_CUDA(cudaMemset(bad,0,sizeof(unsigned long long)));
  const auto start=std::chrono::steady_clock::now();
  auto pass=[&](bool forward){
    for(int cc=0;cc<op.C.ncolors;++cc){
      const int c=forward?cc:(op.C.ncolors-1-cc);
      const int begin=op.C.off[(std::size_t)c],count=op.C.off[(std::size_t)c+1]-begin;
      if(count)m3_block_colour<<<(count+127)/128,128>>>(begin,count,
        op.A.colorRows.p,op.A.row.p,op.A.col.p,op.A.diagPos.p,
        k,op.A.nnz,n,solveRhs,x,op.xy,op.fixed,faceByBubble,op.faces,omega,bad);
    }
  };
  for(int s=0;s<sweeps;++s){pass(true);if(symmetric)pass(false);}
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  unsigned long long failures=0;
  HXT1_CUDA(cudaMemcpy(&failures,bad,sizeof(failures),cudaMemcpyDeviceToHost));
  if(failures)throw std::runtime_error("M3 block diagonal local solve failed count="+std::to_string(failures));
  const double sweepMs=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-start).count();
  if(csrDiagnose){
    const double csrFinal=frozenCsrResidual(solveRhs);
    const double csrRatio=csrFinal/std::max(csrInitial,1.e-300);
    std::printf("NODALS_M3_FROZEN_CSR_RESIDUAL it=%d "
      "convection=%s stress=%s initial=%.12e final=%.12e "
      "ratio=%.12e sweeps=%d omega=%.3f "
      "matrix=EXACT_SAME_AS_SWEEPS measure=TRUE_B_MINUS_KX "
      "status=%s\n",outer,
      op.emacPicard?"emac":"non_emac",
      op.implicitDev?"deviatoric":"vector_laplacian",
      csrInitial,csrFinal,csrRatio,sweeps,omega,
      (std::isfinite(csrInitial)&&std::isfinite(csrFinal) ? "MEASURED":"NONFINITE"));
  }
  if(emacRowL1){
    // Restore saved original diagonal BIT FOR BIT for parity/diagnostics.
    m3_emac_full_row_l1_undo<<<(n+255)/256,256>>>(n,op.A.nnz,
      op.A.diagPos.p,k,rowL1OriginalDiag);
    HXT1_CUDA(cudaGetLastError());
    if(csrDiagnose){
      const double originalFinal=frozenCsrResidual(b);
      std::printf("NODALS_EMAC_M3_ORIGINAL_CSR_RESIDUAL it=%d "
        "initial=%.12e final=%.12e ratio=%.12e "
        "matrix=UNSHIFTED_EMAC_CSR status=MEASURED\n",
        outer,csrInitial,originalFinal,originalFinal/std::max(csrInitial,1.e-300));
    }
  }
  R.its=sweeps;R.cycles=1;R.ok=true;
  R.initial=R.final=R.rel=std::numeric_limits<double>::quiet_NaN();
  std::printf("NODALS_M3_FIXED it=%d method=%s sweeps=%d omega=%.4f "
     "csrNnz=%zu ncolors=%d buildMs=%.3f sweepMs=%.3f "
     "krylov=OFF innerResiduals=NONE bf3=EXACT_TtMT "
     "outerFrozenResidual=EXISTING_DIAGNOSTIC_ONLY status=PASS\n",
     outer,symmetric?"sgs":"fgs",sweeps,omega,op.A.nnz,op.C.ncolors,
     buildMs,sweepMs);
  return R;
}

#include "m4_hybrid_impl.inc"

inline S22D4DResult s22d4d_block_fgmres(const S22D4DOperator&op,
    Real*x,const Real*b,int restart,int maxIts,double rtol,double atol,
    int sweeps,double omega,hxt4a::Reducer<Real>&red,int outer){
  const int n=3*op.A.n;restart=std::min(restart,maxIts);
  auto&W=hxt5e::get_workspace(n,restart);
  S22D4DResult R;
  s22d4d_zero_fixed<<<(n+255)/256,256>>>(op.A.n,op.xy,op.fixed,x);
  if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
    x,x+op.A.n,x+2*(std::size_t)op.A.n);
  R.initial=s22d4d_residual_norm(op,b,x,W,red);R.final=R.initial;
  if(!(atol>=0.)||!std::isfinite(atol))throw std::runtime_error("bad coupled momentum atol");
  const double target=std::max(atol,rtol*R.initial);
  if(!std::isfinite(R.initial))throw std::runtime_error("D4D coupled initial residual nonfinite");
  while(R.its<maxIts && R.final>target){
    ++R.cycles;
    const double beta=s22d4d_residual_norm(op,b,x,W,red);
    hxt5e::copy_scaled_kernel<<<(n+255)/256,256>>>(n,W.r,W.v(0),(Real)(1./beta));
    std::vector<double> H((std::size_t)(restart+1)*(std::size_t)restart,0.),
      cs((std::size_t)restart,0.),sn((std::size_t)restart,0.),g((std::size_t)restart+1,0.);
    g[0]=beta;
    auto h=[&](int i,int j)->double&{return H[(std::size_t)i*(std::size_t)restart+(std::size_t)j];};
    int kdim=0;
    for(int j=0;j<restart && R.its<maxIts;++j){
      op.precondition(W.v(j),W.z(j),sweeps,omega);
      op.apply(W.z(j),W.w);
      for(int i=0;i<=j;++i){
        h(i,j)=red.dot(W.w,W.v(i));
        hxt5e::axpy_kernel<<<(n+255)/256,256>>>(n,(Real)(-h(i,j)),W.v(i),W.w);
      }
      const double hn=red.norm(W.w);
      if(!std::isfinite(hn))throw std::runtime_error("D4D Arnoldi nonfinite");
      h(j+1,j)=hn;
      if(hn>1.e-30)hxt5e::copy_scaled_kernel<<<(n+255)/256,256>>>(n,W.w,W.v(j+1),(Real)(1./hn));
      for(int i=0;i<j;++i){const double a=h(i,j),b0=h(i+1,j);
        h(i,j)=cs[(std::size_t)i]*a+sn[(std::size_t)i]*b0;
        h(i+1,j)=-sn[(std::size_t)i]*a+cs[(std::size_t)i]*b0;
      }
      const double aa=h(j,j),bb=h(j+1,j),rho=std::hypot(aa,bb);
      cs[(std::size_t)j]=rho>0?aa/rho:1.;sn[(std::size_t)j]=rho>0?bb/rho:0.;
      h(j,j)=rho;h(j+1,j)=0.;
      const double gj=g[(std::size_t)j];g[(std::size_t)j]=cs[(std::size_t)j]*gj;
      g[(std::size_t)j+1]=-sn[(std::size_t)j]*gj;
      ++R.its;++kdim;
      if(fabs(g[(std::size_t)j+1])<=target || hn<=1.e-30)break;
    }
    if(kdim<=0)break;
    std::vector<double> y((std::size_t)kdim,0.);
    for(int i=kdim-1;i>=0;--i){double q=g[(std::size_t)i];
      for(int j=i+1;j<kdim;++j)q-=h(i,j)*y[(std::size_t)j];
      if(!(fabs(h(i,i))>1.e-300))throw std::runtime_error("D4D singular Hessenberg");
      y[(std::size_t)i]=q/h(i,i);
    }
    for(int j=0;j<kdim;++j)
      hxt5e::axpy_kernel<<<(n+255)/256,256>>>(n,(Real)y[(std::size_t)j],W.z(j),x);
    s22d4d_zero_fixed<<<(n+255)/256,256>>>(op.A.n,op.xy,op.fixed,x);
    if(op.bf3FluxMode)s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
      x,x+op.A.n,x+2*(std::size_t)op.A.n);
    R.final=s22d4d_residual_norm(op,b,x,W,red);
    if(!std::isfinite(R.final))throw std::runtime_error("D4D true residual nonfinite");
  }
  R.rel=R.final/std::max(R.initial,1.e-300);R.ok=R.final<=target;
  std::printf("STAGE22D4D_COUPLED_KRYLOV it=%d its=%d cycles=%d restart=%d initial=%.12e trueFinal=%.12e trueRel=%.12e rtol=%.4e atol=%.4e converged=%d status=%s\n",
      outer,R.its,R.cycles,restart,R.initial,R.final,R.rel,rtol,atol,(int)R.ok,R.ok?"PASS":"WARN");
  return R;
}
