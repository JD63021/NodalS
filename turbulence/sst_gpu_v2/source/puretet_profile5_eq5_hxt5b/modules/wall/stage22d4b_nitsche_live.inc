// Stage22D.4B: experimentally dispatched Nitsche + Spalding GPU face assembly.
// Disabled by default; enabled only by explicit SST22D4_NITSCHE_ENABLE=1.
// This independent path returns separate scalar blocks and the lagged
// off-component action. Full normal Nitsche uses PHYSICAL wall trace + dn;
// tangential Robin uses physical wall traces, and beta is derived from
// the five normal-ray Spalding samples (as in Stage22D2).
//
// FULL local operator: A_ij(a,b) = n_i n_j [
//    -nu phi_a dn_b - nu dn_a phi_b + gamma nu/h phi_a phi_b]
//    + beta(delta_ij-n_i n_j) phi_a phi_b.
// Entries i==j are assembled into three scalar CSRs; i!=j are applied
// explicitly to lagged velocity in RHS. No block matrix is allocated.
// F1A: two compile-time specializations. The legacy specialization preserves
// the original 12-quadrature direct atomics; the opt-in specialization uses
// per-face shared FP64 reduction without changing local weak-form entries.
// F2: single area-weighted representative Spalding fit per physical wall face.
// This new closure changes ONLY the momentum wall-friction coefficient.
// It retains all twelve quadrature points for viscous/normal integration,
// unchanged Omega wall treatment, original P1+BF3 normal constraint, and
// unchanged frozen-momentum FGMRES operator. No global least squares.
// The input ray points are the existing certified per-quadrature samples.
// We average the VECTOR velocities (not their magnitudes) and sample heights
// with the physical face-quadrature weights, then fit once per face.
// Output: [beta, uTau, yPlus] at three doubles per physical wall face.
__global__ void f2_face_averaged_spalding(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const hxt5b::TetWallPackedPoint*mp,
    const nodals_hxt1::TetVel*tv,
    const Real*u0,const Real*u1,const Real*u2,
    double nu,double*faceFit,unsigned long long*bad){
  const int fi=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(fi>=nf)return;
  using V=::nodals::wall::vector3::V3;
  const auto&f=F[fi];
  const auto&v=tv[f.cell];
  const V n{f.n[0],f.n[1],f.n[2]};
  const unsigned mask=::nodals::wall::active::active_slot_mask(::stage24b_active_points);
  V samples[5]{};double yy[5]{};
  V pic{};
  double areaSum=0.;bool valid=(mp!=nullptr && f.area>0. && nu>0.);
  if(valid){
    for(int q=0;q<12;++q){
      const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
      const double w=ca.w;
      if(!(w>0. && isfinite(w))){valid=false;break;}
      areaSum+=w;
      for(int b=0;b<8;++b){
        const int ib=v.g[b];const double z=w*ca.bp[b];
        pic.x+=z*(double)u0[ib];pic.y+=z*(double)u1[ib];pic.z+=z*(double)u2[ib];
      }
      for(int m=0;m<5;++m){
        if(!(mask & (1u<<m)))continue;
        const auto a=stage27_s02b_at(mp,fi,q,m);
        if(!(a.y>0. && isfinite(a.y))){valid=false;break;}
        yy[m]+=w*a.y;
        for(int b=0;b<8;++b){
          const int ib=a.g[b];const double z=w*a.b[b];
          samples[m].x+=z*(double)u0[ib];
          samples[m].y+=z*(double)u1[ib];
          samples[m].z+=z*(double)u2[ib];
        }
      }
      if(!valid)break;
    }
  }
  ::nodals::wall::vector3::Result fit{};
  if(valid && areaSum>0. && isfinite(areaSum)){
    const double inv=1./areaSum;
    pic.x*=inv;pic.y*=inv;pic.z*=inv;
    for(int m=0;m<5;++m){
      if(mask & (1u<<m)){
        samples[m].x*=inv;samples[m].y*=inv;samples[m].z*=inv;
        yy[m]*=inv;
      }else yy[m]=hxt5b::stage27_s02b_y(m);
    }
    fit=::nodals::wall::vector3::fit_active(
        samples,yy,n,pic,V{0.,0.,0.},nu,::stage24b_active_points);
  }
  if(!fit.valid || !(fit.beta>0.) || !isfinite(fit.beta) ||
      !(fit.uTau>=0.) || !isfinite(fit.yPlus)){
    const unsigned long long prev=atomicAdd(bad,1ULL);
    if(prev<16ULL)printf("NODALS_F2_INVALID face=%d areaSum=%.17g beta=%.17g "
                         "uTau=%.17g yPlus=%.17g activeMask=%u valid=%d\n",
                         fi,areaSum,fit.beta,fit.uTau,fit.yPlus,mask,(int)fit.valid);
    faceFit[(std::size_t)3*fi+0]=-1.;
    faceFit[(std::size_t)3*fi+1]=0.;
    faceFit[(std::size_t)3*fi+2]=0.;
    return;
  }
  faceFit[(std::size_t)3*fi+0]=fit.beta;
  faceFit[(std::size_t)3*fi+1]=fit.uTau;
  faceFit[(std::size_t)3*fi+2]=fit.yPlus;
}

template<bool F1FaceLocal>
__global__ void s22d4_nitsche_tet_component(
    const hxt5b::TetFacePlan*F,int nf,const hxt5b::TetWallQuadCache*qc,
    const hxt5b::TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,const nodals_hxt1::Point*pts,
    const nodals_hxt1::TetConn*tc,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,Real*Ax,Real*Ay,Real*Az,Real*rx,Real*ry,Real*rz,
    double nu,double radius,double kFloor,double omegaFloor,
    double gammaN,unsigned long long*bad,double*frozenCoeff,bool bf3FluxMode,
    Real*penX,Real*penY,Real*penZ,double*stage27ProductionStats,
    bool sampledConductance,
    const ::nodals::wall::shared27::Record*stage27Saved,
    unsigned char*stage27ReuseFlags,
    const double*f2FaceFit
#if defined(NODALS_P2_STAGE4_COUPLED)
    ,const nodals::p2::Tet10Device *p2cells,const Real *p2k,const Real *p2w
#endif
    )
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];const auto tvv=tv[f.cell];const auto g=tg[f.cell];
  if(!(nu>0&&gammaN>=0&&f.area>0&&g.volume>0)){
    if(threadIdx.x==0){
      const unsigned long long prev=atomicAdd(bad,1ULL);
      if(prev<32ULL)printf("STAGE27_NITSCHE_INVALID reason=FACE face=%d owner=%d "
        "nu=%.17g gammaN=%.17g area=%.17g volume=%.17g\n",
        fi,f.cell,nu,gammaN,f.area,g.volume);
    }
    return;
  }
  const double h=3.*g.volume/f.area;
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const Real* us[3]={u0,u1,u2};Real* mats[3]={Ax,Ay,Az};Real* rhs[3]={rx,ry,rz};
  Real* penMats[3]={penX,penY,penZ};
  // 12 quadrature contributions, 64 local basis pairs, 3 components.
  // Shared memory: 39,168 bytes in the F1A specialization (below 48 KiB).
  // The unused arrays are eliminated in the legacy template instantiation.
  __shared__ double f1Matrix[12][64][3];
  __shared__ double f1Penalty[12][64][3];
  __shared__ double f1Rhs[12][8][3];
  if constexpr(F1FaceLocal){
    for(int z=(int)threadIdx.x;z<12*64*3;z+=(int)blockDim.x){
      ((double*)f1Matrix)[z]=0.;
      ((double*)f1Penalty)[z]=0.;
    }
    for(int z=(int)threadIdx.x;z<12*8*3;z+=(int)blockDim.x)
      ((double*)f1Rhs)[z]=0.;
    __syncthreads();
  }
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto &ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    if(!(ca.w>0.&&mp)){
      const unsigned long long prev=atomicAdd(bad,1ULL);
      if(prev<32ULL)printf("STAGE27_NITSCHE_INVALID reason=QUAD face=%d q=%d owner=%d "
        "weight=%.17g rayPointer=%d\n",fi,q,f.cell,ca.w,(int)(mp!=nullptr));
      continue;
    }
    using V=::nodals::wall::vector3::V3;
    ::nodals::wall::vector3::Result fit{};
    bool stage27Reused=false;
    const std::size_t stage27Index=(std::size_t)fi*12u+(std::size_t)q;
    if(f2FaceFit){
      fit.beta=f2FaceFit[(std::size_t)3*fi];
      fit.uTau=f2FaceFit[(std::size_t)3*fi+1u];
      fit.yPlus=f2FaceFit[(std::size_t)3*fi+2u];
      fit.valid=(fit.beta>0. && isfinite(fit.beta) && isfinite(fit.uTau) && isfinite(fit.yPlus));
    }else{
      const V vn{n[0],n[1],n[2]};
      V samples[5]{};double yy[5];
      const unsigned stage27ActiveMask=
          ::nodals::wall::active::active_slot_mask(::stage24b_active_points);
      for(int m=0;m<5;++m){
        const auto &a=stage27_s02b_at(mp,fi,q,m);
        yy[m]=(stage27ActiveMask&(1u<<m))?a.y:hxt5b::stage27_s02b_y(m);
        if(!(stage27ActiveMask & (1u<<m)))continue;
        V s{0,0,0};
        for(int b=0;b<8;++b){
          const int ib=a.g[b];const double weight=a.b[b];
          s.x+=weight*(double)u0[ib];s.y+=weight*(double)u1[ib];s.z+=weight*(double)u2[ib];
        }
        samples[m]=s;
      }
      V pic{0,0,0};
      for(int b=0;b<8;++b){
        const int ib=tvv.g[b];const double weight=ca.bp[b];
        pic.x+=weight*(double)u0[ib];pic.y+=weight*(double)u1[ib];pic.z+=weight*(double)u2[ib];
      }
      fit=::nodals::wall::shared27::fit_or_reuse(
          samples,yy,vn,pic,V{0.,0.,0.},nu,::stage24b_active_points,
          stage27Saved?stage27Saved+stage27Index:nullptr,&stage27Reused);
    }
    if(stage27ReuseFlags)stage27ReuseFlags[stage27Index]=stage27Reused?1u:0u;
    if(!fit.valid || !(fit.beta>=0) || !isfinite(fit.beta)){
      const unsigned long long prev=atomicAdd(bad,1ULL);
      if(prev<32ULL)printf("STAGE27_NITSCHE_INVALID reason=SPALDING face=%d q=%d "
        "mode=%s valid=%d beta=%.17g utau=%.17g yp=%.17g reused=%d nu=%.17g\n",
        fi,q,f2FaceFit?"F2_FACE_MEAN":"LEGACY_QUAD",(int)fit.valid,
        fit.beta,fit.uTau,fit.yPlus,(int)stage27Reused,nu);
      continue;
    }
    // Stage 1.A: piggyback on the already-required Spalding fit.
    // One fixed pre-momentum field state per SIMPLE iteration.
    // No extra ray tracing, velocity reconstruction or nonlinear solve.
    if(stage27ProductionStats){
      atomicAdd(stage27ProductionStats+0,ca.w);
      atomicAdd(stage27ProductionStats+1,ca.w*fit.uTau*fit.uTau);
      atomicAdd(stage27ProductionStats+2,ca.w*fit.yPlus);
    }
    // Fill affine barycentrics from the constant device face topology.
    double lam[4]={0.,0.,0.};
    lam[hxt2::c_tet_face[(int)f.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)f.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)f.lf][2]]=hxt5b::QL2[q];
    // Use the SAME constitutive provider as volume SST momentum diffusion.
    // Both physical boundary traces and normal gradients include all 8 P1+BF3 DOFs.
    double ph[8],gr[8][3],gp[8][3],dn[8];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,g,gp);
    Real gu[9]={Real(0)},gk[3]={Real(0)},gw[3]={Real(0)},kv=Real(0),wv=Real(0);
    double x=0.,y=0.;
    for(int a=0;a<4;++a){const auto P=pts[tc[f.cell].v[a]];x+=lam[a]*P.x;y+=lam[a]*P.y;}
    for(int a=0;a<8;++a){
      const int ia=tvv.g[a];
#if defined(NODALS_P2_STAGE4_COUPLED)
      if(!p2cells){
#endif
        kv+=(Real)ph[a]*k[ia];wv+=(Real)ph[a]*omega[ia];
#if defined(NODALS_P2_STAGE4_COUPLED)
      }
#endif
      dn[a]=gp[a][0]*n[0]+gp[a][1]*n[1]+gp[a][2]*n[2];
      for(int d=0;d<3;++d){
        const Real gd=(Real)gp[a][d];
        gu[d]+=u0[ia]*gd;gu[3+d]+=u1[ia]*gd;gu[6+d]+=u2[ia]*gd;
#if defined(NODALS_P2_STAGE4_COUPLED)
        if(!p2cells){
#endif
          gk[d]+=k[ia]*gd;gw[d]+=omega[ia]*gd;
#if defined(NODALS_P2_STAGE4_COUPLED)
        }
#endif
      }
    }
#if defined(NODALS_P2_STAGE4_COUPLED)
    if(p2cells){
      double pb[10],pr[10][3],pg[10][3];
      nodals::p2::basis(lam,pb,pr);
      nodals::p2::physical_gradient(pr,g.invJ,pg);
      for(int a=0;a<10;++a){
        const int ia=p2cells[f.cell].g[a];
        const Real kc=p2k[ia],wc=p2w[ia];
        kv+=kc*(Real)pb[a];wv+=wc*(Real)pb[a];
        for(int d=0;d<3;++d){gk[d]+=kc*(Real)pg[a][d];gw[d]+=wc*(Real)pg[a][d];}
      }
    }
#endif
    const auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
       gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,
       hxt5c::stage20_distance(tc,f.cell,lam));
    const double nueff=nu+(double)ss.nut;
    if(!(nueff>0.)||!isfinite(nueff)){
      const unsigned long long prev=atomicAdd(bad,1ULL);
      if(prev<32ULL)printf("STAGE27_NITSCHE_INVALID reason=VISCOSITY face=%d q=%d "
        "owner=%d nu=%.17g nut=%.17g nueff=%.17g kval=%.17g "
        "omega=%.17g kfloor=%.17g omegafloor=%.17g\n",
        fi,q,f.cell,nu,(double)ss.nut,nueff,(double)kv,(double)wv,
        kFloor,omegaFloor);
      continue;
    }
    if(frozenCoeff){
      const std::size_t iq=(std::size_t)fi*12u+(std::size_t)q;
      frozenCoeff[2u*iq]=nueff;frozenCoeff[2u*iq+1u]=fit.beta;
    }
    // Stage3.A: each quadrature thread owns one (face,q). Gather its
    // off-component lagged wall RHS locally before a single global write
    // per component and test function. The same nine tensor coefficients,
    // element stiffness values, and BF3 constraints are preserved.
    for(int a=0;a<8;++a){
      const int ia=tvv.g[a];const double pa=ca.bw[a],dna=dn[a];
      double stage27_s03a_rhs[3]={0.,0.,0.};
      for(int b=0;b<8;++b){
        const int ib=tvv.g[b];
        const double pb=ca.bw[b],dnb=dn[b];
        const double normalQ=bf3FluxMode ? ca.w*nueff*(gammaN/h*pa*pb) :
  ca.w*nueff*(-pa*dnb-dna*pb+gammaN/h*pa*pb);
        // Opt-in: tangential friction acts on the SAME normal-column Picard
        // functional used by the frozen beta fit, not on the wall trace.
        // The normal BF3 constraint and penalty still use wall traces.
        const double robin=ca.w*fit.beta*
          (sampledConductance?ca.bp[a]*ca.bp[b]:pa*pb);
        const int position=slot[(std::size_t)f.cell*64u+(std::size_t)(a*8+b)];
        // Stage3.B: the exact normal/tangential tensor is a diagonal
        // Robin part plus the rank-one normal coupling:
        // A_ij = delta_ij*robin + n_i*n_j*(normalQ-robin).
        // The momentum offdiagonal action is still lagged, as in Stage3.A.
        // We factor its six coefficients without ever omitting a component.
        const double stage27_s03b_coupling=normalQ-robin;
        const double stage27_s03b_other[3]={
          n[1]*(double)us[1][ib]+n[2]*(double)us[2][ib],
          n[0]*(double)us[0][ib]+n[2]*(double)us[2][ib],
          n[0]*(double)us[0][ib]+n[1]*(double)us[1][ib]};
        for(int i=0;i<3;++i){
          const double nii=n[i],nii2=nii*nii;
          const double entry=nii2*normalQ+(1.-nii2)*robin;
          if constexpr(F1FaceLocal)
            f1Matrix[q][a*8+b][i]=entry;
          else
            atomicAdd(mats[i]+position,(Real)entry);
          if(bf3FluxMode && penMats[i]){
            const double pen=nii2*ca.w*nueff*(gammaN/h)*pa*pb;
            if constexpr(F1FaceLocal)
              f1Penalty[q][a*8+b][i]=pen;
            else
              atomicAdd(penMats[i]+position,(Real)pen);
          }
          stage27_s03a_rhs[i]+=-(nii*stage27_s03b_coupling)*stage27_s03b_other[i];
        }
      }
      for(int i=0;i<3;++i){
        if constexpr(F1FaceLocal)
          f1Rhs[q][a][i]=stage27_s03a_rhs[i];
        else
          atomicAdd(rhs[i]+ia,(Real)stage27_s03a_rhs[i]);
      }
    }
  }
  if constexpr(F1FaceLocal){
    // All 12 quadrature lanes have completed their contributions.  Invalid
    // quadrature points leave the associated shared entries at zero, just as
    // the legacy kernel skips those entries.
    __syncthreads();
    for(int pair=(int)threadIdx.x;pair<64;pair+=(int)blockDim.x){
      const int a=pair/8,b=pair%8;
      const int position=slot[(std::size_t)f.cell*64u+(std::size_t)pair];
      for(int i=0;i<3;++i){
        double sum=0.,psum=0.;
        for(int q=0;q<12;++q){
          sum+=f1Matrix[q][pair][i];
          if(bf3FluxMode && penMats[i])psum+=f1Penalty[q][pair][i];
        }
        atomicAdd(mats[i]+position,(Real)sum);
        if(bf3FluxMode && penMats[i])
          atomicAdd(penMats[i]+position,(Real)psum);
      }
      (void)a;(void)b;
    }
    for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){
      const int ia=tvv.g[a];
      for(int i=0;i<3;++i){
        double sum=0.;
        for(int q=0;q<12;++q)sum+=f1Rhs[q][a][i];
        atomicAdd(rhs[i]+ia,(Real)sum);
      }
    }
  }
}

// Post-pressure-correction wall flux on the DISCRETE triangulated geometry.
// Signed and absolute normal velocities are integrated with the physical
// boundary test trace, not a reconstructed CAD normal.
__global__ void s22d4b_wall_normal_audit(const hxt5b::TetFacePlan*F,int nf,
  const hxt5b::TetWallQuadCache*qc,const nodals_hxt1::TetVel*tv,
  const Real*u0,const Real*u1,const Real*u2,double*stats){
  int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto f=F[fi];const auto tvv=tv[f.cell];
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    double vn=0.;
    for(int a=0;a<8;++a){
      const int ia=tvv.g[a];const double w=ca.bw[a];
      vn+=w*((double)u0[ia]*f.n[0]+(double)u1[ia]*f.n[1]+(double)u2[ia]*f.n[2]);
    }
    atomicAdd(stats+0,ca.w);atomicAdd(stats+1,ca.w*fabs(vn));
    atomicAdd(stats+2,ca.w*vn);atomicAdd(stats+3,ca.w*vn*vn);
  }
}
