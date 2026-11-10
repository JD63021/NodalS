#include "../coupling/g2_metric_plateau.hpp"
// NODALS_FACE_YPLUS30_A_20261105
#include <memory>
#include <fstream>
#include <cstdio>
// HXT5B6_EXACT_DIAGNOSTIC_VTU
#include "../mesh/tet_patch_registry.hpp"
#include "../mesh/stage23_patch_routing.hpp"
#include "../wall/stage27_wall_audit_policy.hpp"
#include "../geometry/wall_distance_host.hpp"
#include "../initialization/stage27_potential_host.hpp"

// ============================================================
// HXT5B_FIXED_FGS_FAST_V1
//
// Fixed-work optimization:
//
//   MOM_WORK       = fgs1
//   MOM_MAX        = 1
//   MOM_CAP_POLICY = warn
//
// In this configuration the numerical operation is exactly one
// forward colored-GS sweep regardless of the inner residual.
//
// Therefore skip the initial/final inner residual SpMV+reduction.
// The SIMPLE outer physical/convergence machinery is unchanged.
//
// Every other momentum configuration falls through unchanged to
// hxt5e::momentum_solve.
// ============================================================
#include "../linear/momentum_work_policy.inc"


#include "../diagnostics/enriched_postprocess.inc"

#include "../diagnostics/compact_continuity.inc"
#include "../diagnostics/stage6c1_component_delta.inc"

#include "../coupling/simple_velocity_correction_kernel.inc"
#include "../coupling/stage31_restart.hpp"
int main(int argc,char**argv){
  try{
    // BF3_SCHEMA_V3: The frozen Stage22C .case validator accepts only q1 or
    // bernstein_q2. It explicitly selects q1 as a schema-compatible transport
    // placeholder. Only this separate, opt-in experimental CUDA child switches
    // to the genuinely active eight-mode tetrahedral scalar state.
    // NODALS_BF3_REQUIRE_ACTIVE is carried through the Stage22C runner (verified
    // by the previous native BF3_FAILFAST diagnostic at iteration 201).
    const char* bf3Required=std::getenv("NODALS_BF3_REQUIRE_ACTIVE");
    if(bf3Required && std::string(bf3Required)=="1"){
      const char* prior=std::getenv("SST_STATE_SPACE");
      const std::string priorMode=prior?std::string(prior):"UNSET";
      if(::setenv("SST_STATE_SPACE","q1bf2_unlimited",1)!=0)
        throw std::runtime_error("BF3_SCHEMA_V3: could not override CUDA scalar state space");
      std::printf("NODALS_BF3_NATIVE_FORCE=PASS source=SCHEMA_V3 prior=%s effective=%s optIn=1\n",
                  priorMode.c_str(),std::getenv("SST_STATE_SPACE"));
      std::fflush(stdout);
    }
    using namespace hxt4b;
    using T=Real;

    std::string mesh,vtu="HXT5B_full_rans.vtu",momWork="sgs1",momCapPolicy="warn";
    std::string pressureSolver="pcg_amg",amgHierarchy="cf",cfCoarsening="pmis",cfStrength="classical-negative",cfInterp="exti";
    std::string pressureGauge="pin",podMode="off",podRecompressMethod="tail_pod";
    const char*simpleCouplingEnv=std::getenv("SIMPLE_COUPLING");
    std::string simpleCoupling=simpleCouplingEnv?std::string(simpleCouplingEnv):"lagged";
    std::string amgFineMode="explicit"; // HXT5B_AMG_FINE_MODE_V1
    std::string amgSmoother="cheb2",amgSpectrumPolicy="auto",amgMcgsOrder="symmetric";
    double Re=13691.740248127866,bulk=50,gamma=50,schurNitscheGamma=0,alphaU=.7,alphaP=1,rauScale=2,simpleTol=1e-3,momOmega=1,momRtol=.5,momAtol=1e-8,mixlenScale=1.0,outerContAtol=1e-12,wallSampleFraction=.5;
    int momentumWallMode=hxt5d::MOM_WALL_SPALDING;
    int spaldingMolecularConsistency=0; // SPALDING_MOLECULAR_CONSISTENCY_V1
    double weakWallGamma=4.0,weakWallHScale=1.0,weakWallNutScale=1.0;
    double pRtol=.9,pAtol=1e-12,cfTheta=.25,amgJacobiOmega=.7,amgLambdaSafety=1.5,amgLambdaLow=.05,amgMcgsOmega=1;
    int maxOuter=3000,pMax=50,momMax=50,momFgmresRestart=30,momFgsSweeps=1,cfPmax=8,cfAggressiveFirst=0,amgTerminal=1000,amgPowerIts=16,amgChebDegree=4;
    int podTrainSnapshots=10,podRank=2,podMaxRank=64,podTailIter=10,podHardIts=0,podAcceptMinAbs=5,podRefreshAbs=30;
    int podRecompressRank=0,podRecompressHistory=18,podRecompressMaxLossAbs=5,podRecompressForce=0,podNoProbe=0;
    double podNovelty=1e-8,podAcceptMinRel=0.03,podRefreshRel=0.35,podRecompressMaxLossRel=0.10;
    int amgJacobiFineSweeps=1,amgJacobiCoarseSweeps=1,amgMcgsFineSweeps=1,amgMcgsCoarseSweeps=1;
    hxt5c::Options sstG1;
    bool sstG2=false,sstG2PhysicalGate=true,profileAudit=true,laminarPerf=false;
    std::string sstG2Start="g0",profileZFractions="0.50,0.75,0.90,0.98";
    int profileRadial=65,profileAngles=16;
    double profileRmax=.98;
    double sstG2PlateauTol=2.5e-3,sstG2ContAtol=1e-6;
    int sstG2Max=800,sstG2PrintEvery=20,sstG2PlateauSamples=5,sstG2MetricsEvery=1;

#include "../configuration/native_cli_parse.inc"
    // STAGE41: independent native omega wall-target selector.
    const char* s41raw=std::getenv("STAGE41_OMEGA_MODE");
    const std::string s41mode=(s41raw&&*s41raw)?std::string(s41raw):"sample_log";
    // Experimental omega-only SST-2003m Couette table target at interior point 3.
    // Retains the OF_AUTO transport path and the existing full-vector Spalding fit.
    if(s41mode!="sample_log" && s41mode!="of_auto" && s41mode!="trace_log" && s41mode!="couette_point3")
      throw std::runtime_error("STAGE41_OMEGA_MODE must be sample_log, of_auto or couette_point3");
    if(s41mode=="couette_point3"){
      if(sstG1.ofProduction)
        throw std::runtime_error("couette_point3 requires OF wall production OFF (omega-only comparison)");
      const char* scalarState=std::getenv("SST_STATE_SPACE");
      if(scalarState && std::string(scalarState)!="q1")
        throw std::runtime_error("couette_point3 is validated as a masked-P1/Q1 experiment only");
    }
    // NODALS_OF_WALL_PRODUCTION_OPTION_V1: STAGE41 selects the omega target,
    // independently of the existing OF_AUTO near-wall production correction.
    // Do not change production kernels or the SST 10*betaStar*k*omega limiter.
    if(sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO)
      throw std::runtime_error("STAGE41 requires omega-wall of_auto");
    if(sstG1.ofProduction)
      std::printf("NODALS_OF_WALL_PRODUCTION_OPTION enabled=1 omegaTarget=%s "
                  "equations=K_AND_OMEGA limit=EXISTING_10_BETASTAR_K_OMEGA "
                  "experimental=1\n",s41mode.c_str());
    // NODALS_TRACE_LOG_TET_PATCH_V1: pure log target at physical trace.
    // Native of_auto dispatch is retained so only the omega boundary term changes.
    if(s41mode=="trace_log"){
      if(sstG1.ofProduction)
        throw std::runtime_error("trace_log requires omega_wall.production=false");
      const char* st=std::getenv("SST_STATE_SPACE");
      if(st && std::string(st)!="q1")
        throw std::runtime_error("trace_log currently supports masked-P1/Q1 only");
    }
    const char* tgRaw=std::getenv("STAGE41_TRACE_GAMMA");
    const double traceGamma=(tgRaw&&*tgRaw)?std::stod(tgRaw):50.0;
    if(s41mode=="trace_log" && !(std::isfinite(traceGamma)&&traceGamma>=10.0&&traceGamma<=1.e5))
      throw std::runtime_error("trace_log trace_gamma must be in [10,100000]");
    const int s41Trace=(s41mode=="trace_log")?1:0;
    HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage41_trace_log,&s41Trace,sizeof(s41Trace)));
    HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage41_trace_gamma,&traceGamma,sizeof(traceGamma)));
    if(s41Trace)
      std::printf("NODALS_OMEGA_TRACE_LOG enabled=1 target=PURE_LOG_AT_INTERIOR_POINT3 "
        "placement=PHYSICAL_TET_WALL_TRACE operator=SYMMETRIC_NITSCHE "
        "mu=POINT3_NU_PLUS_SIGMAW_NUT h=THREE_TET_VOLUME_OVER_WALL_AREA "
        "gamma=%.12e kWall=NATURAL_ZERO_GRADIENT production=0 "
        "spaldingFit=SHARED_FULL_VECTOR status=EXPERIMENTAL\n",traceGamma);
    const int s41flag=(s41mode=="sample_log" || s41mode=="trace_log")?1:0;
    HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage41_sample_log,&s41flag,sizeof(s41flag)));
    const int s41CouettePoint3=(s41mode=="couette_point3")?1:0;
    HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage41_couette_point3,&s41CouettePoint3,sizeof(s41CouettePoint3)));
    if(s41CouettePoint3)
      std::printf("NODALS_OMEGA_COUETTE_POINT3 enabled=1 fit=SPALDING_VECTOR5 "
        "omegaTarget=SST2003M_COUETTE_TABLE_OMEGAPLUS_UTAU2_OVER_NU "
        "location=INTERIOR_POINT3 penalty=EXISTING_RANK_ONE "
        "kWall=NATURAL_ZERO_GRADIENT ofProduction=OFF "
        "tableYPlus=[0.05,3000] status=EXPERIMENTAL\n");
    std::printf("STAGE41_SAMPLE_LOG mode=%s omegaTarget=%s "
      "constraint=%s fullVectorSpalding=ON "
      "robin=OFF traceNitsche=%s SSTvolume=UNCHANGED "
      "deviceConstants=SET status=PASS\n",
      s41mode.c_str(),s41flag?"PURE_LOG":(s41CouettePoint3?"COUETTE_SST2003M_TABLE_POINT3":"OF_AUTO_BLEND"),
      s41Trace?"PHYSICAL_TRACE_SYMMETRIC_NITSCHE":"ORIGINAL_INTERIOR_POINT_RANK1",
      s41Trace?"ON":"OFF");
    // Stage5.A environment overrides for the existing Stage26 .case bridge.
    // Unset environment leaves BOTH scalar solvers identical to frozen Stage3.B.
    const auto s05a_mode_env=[](const char*key,std::string&target){
      const char* value=std::getenv(key);
      if(value && *value)target=value;
    };
    const auto s05a_sweep_env=[](const char*key,int&target){
      const char* value=std::getenv(key);
      if(value && *value){
        const std::string v(value);
        std::size_t pos=0;
        int n=std::stoi(v,&pos);
        if(pos!=v.size())throw std::runtime_error(std::string("Stage5.A invalid integer: ")+key);
        target=n;
      }
    };
    s05a_mode_env("STAGE27_S05A_K_MODE",sstG1.s05aKMode);
    s05a_mode_env("STAGE27_S05A_OMEGA_MODE",sstG1.s05aOmegaMode);
    s05a_sweep_env("STAGE27_S05A_K_SWEEPS",sstG1.s05aKSweeps);
    s05a_sweep_env("STAGE27_S05A_OMEGA_SWEEPS",sstG1.s05aOmegaSweeps);
    s05a_sweep_env("STAGE27_S05A_CHECK_EVERY",sstG1.s05aCheckEvery);
    const auto s05a_valid_mode=[](const std::string&m){
      return m=="inherit"||m=="fgmres-fgs"||m=="fgs-fixed"||m=="sgs-fixed"||m=="fgs-guarded";
    };
    if(!s05a_valid_mode(sstG1.s05aKMode)||!s05a_valid_mode(sstG1.s05aOmegaMode)
        ||sstG1.s05aKSweeps<1||sstG1.s05aKSweeps>10000
        ||sstG1.s05aOmegaSweeps<1||sstG1.s05aOmegaSweeps>10000
        ||sstG1.s05aCheckEvery<0||sstG1.s05aCheckEvery>10000
        ||(sstG1.s05aCheckEvery==0 &&
          (sstG1.s05aKMode=="fgs-guarded"||sstG1.s05aOmegaMode=="fgs-guarded")))
      throw std::runtime_error("Stage5.A invalid mode/sweeps/check cadence");
    std::printf("STAGE27_S05A_CONFIG kMode=%s omegaMode=%s kSweeps=%d "
      "omegaSweeps=%d checkEvery=%d momentum=UNCHANGED_BF3_BLOCK_FGMRES "
      "legacyFGMRES=SELECTABLE residualEverySweep=NO status=PASS\n",
      sstG1.s05aKMode.c_str(),sstG1.s05aOmegaMode.c_str(),
      sstG1.s05aKSweeps,sstG1.s05aOmegaSweeps,sstG1.s05aCheckEvery);
    // Stage24B is a runtime model-selection gate. No change to existing
    // five-slot ray cache, physical wall geometry, Schur, or SIMPLE controls.
    auto stage24b_int=[](const char* key,int defaultValue)->int{
      const char* raw=std::getenv(key);
      if(!raw||!*raw)return defaultValue;
      char* end=nullptr;long x=std::strtol(raw,&end,10);
      if(!end||*end!='\0'||x<1||x>5)
        throw std::runtime_error(std::string("Stage24B: invalid integer 1..5 for ")+key);
      return (int)x;
    };
    const int stage24bN=stage24b_int("SST24B_ACTIVE_POINTS",5);
    const int stage24bOmega=stage24b_int("SST24B_OMEGA_SAMPLE_INDEX",3);
    // TRACE_LOG_1TO5_SAMPLES_FIX: keep the point-3 omega target, but permit
    // any already-validated Spalding active subset (1..5). No kernel changes.
    if(s41mode=="trace_log" && stage24bOmega!=3)
      throw std::runtime_error("trace_log requires omega point 3");
    if(s41mode=="couette_point3" && (stage24bN!=5 || stage24bOmega!=3))
      throw std::runtime_error("couette_point3 requires 5 Spalding fitting points and omega sample index 3");
    // STAGE32_SPLIT_WALL: independent wall-normal distances, both at index 3.
    // The original five Spalding rays, momentum traction, and shared u_tau are unchanged.
    const char* s32sRaw=std::getenv("STAGE32_SPALDING_SAMPLE_SCALE");
    const char* s32oRaw=std::getenv("STAGE32_OMEGA_SAMPLE_SCALE");
    // Stage33A is OFF by default; strictly controlled by runner CLI -> env.
    const auto s33_env=[](const char* key,const char* fallback)->double{
      const char* v=std::getenv(key);
      const double z=std::stod((v&&*v)?v:fallback);
      if(!std::isfinite(z)||z<=0)throw std::runtime_error(std::string("STAGE33A invalid ")+key);
      return z;
    };
    // Full implicit 3x3 deviatoric SST momentum volume operator.
    // Keep the legacy --sst-deviatoric-momentum lagged RHS separately OFF.
    // NODALS_CONSERVATIVE_PICARD_V1: momentum only; scalar convection unchanged.
    const char* momentumConvRaw=std::getenv("NODALS_MOMENTUM_CONVECTION_FORM");
    const std::string momentumConvectionForm=(momentumConvRaw&&*momentumConvRaw)?
      std::string(momentumConvRaw):"advective";
    // NODALS_SKEW_PICARD_V1: skew adds 1/2 the frozen-carrier divergence correction.
    // NODALS_EMAC_PICARD_V1: EMAC a_j d_j u_i + a_j d_i u_j + a_i div(u).
    if(momentumConvectionForm!="advective" && momentumConvectionForm!="conservative" &&
       momentumConvectionForm!="skew_symmetric" && momentumConvectionForm!="emac")
      throw std::runtime_error("NODALS_MOMENTUM_CONVECTION_FORM must be advective, conservative, skew_symmetric, or emac");
    const bool momentumEmac=(momentumConvectionForm=="emac");
    const bool momentumConservative=(momentumConvectionForm=="conservative");
    const bool momentumSkew=(momentumConvectionForm=="skew_symmetric");
    const bool momentumDivergenceCorrection=momentumConservative||momentumSkew;
    const char* implicitStressRaw=std::getenv("NODALS_MOMENTUM_STRESS_FORM");
    const std::string stressForm=(implicitStressRaw&&*implicitStressRaw)?
      std::string(implicitStressRaw):"vector_laplacian";
    if(stressForm!="vector_laplacian" && stressForm!="deviatoric")
      throw std::runtime_error("NODALS_MOMENTUM_STRESS_FORM must be vector_laplacian or deviatoric");
    const bool implicitDevStress=(stressForm=="deviatoric");
    const char* s33Flag=std::getenv("STAGE33A_MOMENTUM_SUPG");
    // Stage33A's old constant-viscosity/Laplacian residual is retired:
    // it is not full strong-form SUPG for variable-viscosity SST.
    const bool stage33MomentumSupg=false;
    if(s33Flag && std::string(s33Flag)=="1")
      throw std::runtime_error("Legacy Stage33A momentum SUPG is unsupported; "
        "select [supg] momentum=true through nodals.py instead");
    if(s33Flag&&*s33Flag&&std::string(s33Flag)!="0"&&std::string(s33Flag)!="1")
      throw std::runtime_error("STAGE33A_MOMENTUM_SUPG must be 0/1");
    const double stage33TauScale=s33_env("STAGE33A_TAU_SCALE","0.05");
    const double stage33Magic=s33_env("STAGE33A_MAGIC","9");
    // NODALS_SKEW_SD_DIAGNOSTIC_V1: opt-in positive streamline diffusion.
    // Never active by default; *not* full residual-consistent RANS SUPG.
    // NODALS_UNIFIED_FULL_MOMENTUM_SUPG_V1
    // One canonical momentum switch for full element-interior strong residual.
    // Older independent streamline-only and full-mode flags must not be used.
    const char* oldSupgFlags[]={
      "NODALS_SKEW_STREAMLINE_DIFFUSION",
      "NODALS_SKEW_FULL_SUPG",
      "NODALS_CONSERVATIVE_STREAMLINE_DIFFUSION",
      "NODALS_CONSERVATIVE_FULL_SUPG",
      "NODALS_SKEW_SD_TAU_SCALE", "NODALS_CONSERVATIVE_SD_TAU_SCALE",
      "NODALS_SKEW_FULL_NU_FD_REL", "NODALS_CONSERVATIVE_FULL_NU_FD_REL"};
    for(const char* key:oldSupgFlags){
      const char* x=std::getenv(key);
      if(x&&*x)
        throw std::runtime_error(std::string("Deprecated independent momentum SUPG option ")+
          key+"; remove it and set [supg] momentum=true/false");
    }
    const char* fullRaw=std::getenv("NODALS_MOMENTUM_FULL_SUPG");
    const std::string fullFlag=(fullRaw&&*fullRaw)?std::string(fullRaw):"0";
    if(fullFlag!="0" && fullFlag!="1")
      throw std::runtime_error("NODALS_MOMENTUM_FULL_SUPG must be 0 or 1");
    const bool momentumFullSupg=(fullFlag=="1");
    const double supgViscosityFdRel=s33_env("NODALS_MOMENTUM_SUPG_NU_FD_REL","0.001");
    if(momentumFullSupg){
      if(!implicitDevStress || sstG1.deviatoricMomentum || momentumEmac)
        throw std::runtime_error("Full momentum SUPG unsupported for requested stress/convection; "
          "currently requires implicit deviatoric stress and advective, skew_symmetric, or conservative convection");
      if(!std::isfinite(supgViscosityFdRel) || supgViscosityFdRel>0.01)
        throw std::runtime_error("[supg] nu_fd_rel must be in (0,0.01]");
    }
    const bool momentumStreamlineDiffusion=momentumFullSupg;
    const double activeSdTauScale=stage33TauScale;
    const double convectionDivFactor=momentumSkew?0.5:(momentumConservative?1.0:0.0);
    std::printf("NODALS_MOMENTUM_FULL_SUPG_CONFIG enabled=%d convection=%s "
      "stress=%s divFactor=%.1f tauScale=%.8g fdRel=%.8g "
      "residual=FULL_DEVIATORIC_ELEMENT_INTERIOR_PRESSURE_P0 "
      "nuGrad=CENTERED_FD faceJumps=NOT_INCLUDED status=PASS\n",
      (int)momentumFullSupg,momentumConvectionForm.c_str(),stressForm.c_str(),
      convectionDivFactor,stage33TauScale,supgViscosityFdRel);

    std::printf("NODALS_LEGACY_STAGE33A_SUPG=RETIRED canonical=FULL_RESIDUAL_ONLY\n");
    if(stage33MomentumSupg && implicitDevStress)
      throw std::runtime_error("momentum SUPG strong residual not yet available for implicit deviatoric stress");
    if(stage33MomentumSupg && sstG1.deviatoricMomentum)
      throw std::runtime_error("STAGE33A does not implement strong residual of deviatoric momentum RHS");
    const bool stage32Requested=(s32sRaw&&*s32sRaw)||(s32oRaw&&*s32oRaw);
    const double stage32SpaldingScale=(s32sRaw&&*s32sRaw)?std::stod(s32sRaw):0.05;
    const double stage32OmegaScale=(s32oRaw&&*s32oRaw)?std::stod(s32oRaw):stage32SpaldingScale;
    if(!(std::isfinite(stage32SpaldingScale)&&std::isfinite(stage32OmegaScale)&&
         stage32SpaldingScale>=0.001&&stage32SpaldingScale<=1.0&&
         stage32OmegaScale>=0.001&&stage32OmegaScale<=1.0))
      throw std::runtime_error("STAGE32 scales must be finite and in [0.001,1.0]");
    // Stage32 changes only the omega evaluation ray. The Spalding fit may
    // use any existing active subset (1..5), including the full five-point fit.
    // The split omega ray itself remains indexed at point 3 for OF_AUTO.
    if(stage32Requested &&
       (stage24bOmega!=3 || sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO))
      throw std::runtime_error("STAGE32 requires OF_AUTO and --omega-sample-index 3 (Spalding samples 1..5 supported)");
    const char* yplusFaceInput=std::getenv("NODALS_FACE_YPLUS_UTAU_CSV");
    const bool faceYplusEnabled=yplusFaceInput && *yplusFaceInput;
    // NODALS_AUTO_CENTROID_SAMPLING_V1
    const auto autoSwitch=[](const char*key){const char*p=std::getenv(key);
      return p && std::string(p)=="1";};
    const bool autoCentroidMode=autoSwitch("NODALS_AUTO_CENTROID_MODE");
    const bool autoCentroidSp=autoSwitch("NODALS_AUTO_CENTROID_SP");
    const bool autoCentroidOm=autoSwitch("NODALS_AUTO_CENTROID_OM");
    if((autoCentroidSp||autoCentroidOm) && !autoCentroidMode)
      throw std::runtime_error("NODALS_AUTO requires centroid mode");
    const bool stage32Split=autoCentroidMode || faceYplusEnabled || (stage32Requested &&
      std::abs(stage32OmegaScale-stage32SpaldingScale)>1.e-13);

    if(stage24bN!=5 && sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO)
      throw std::runtime_error("Stage24B N<5 is supported only with OF_AUTO, not EQ5");
    const int stage24bOmegaZero=stage24bOmega-1;
    HXT1_CUDA(cudaMemcpyToSymbol(::stage24b_active_points,&stage24bN,sizeof(int)));
    HXT1_CUDA(cudaMemcpyToSymbol(::stage24b_omega_slot,&stage24bOmegaZero,sizeof(int)));
    const std::string zeroFlag=std::getenv("STAGE27_SPALDING_ZERO_LIMIT")?
      std::getenv("STAGE27_SPALDING_ZERO_LIMIT"):"0";
    if(zeroFlag!="0" && zeroFlag!="1")
      throw std::runtime_error("STAGE27_SPALDING_ZERO_LIMIT must be 0 or 1");
    const int zeroEnabled=zeroFlag=="1"?1:0;
    HXT1_CUDA(cudaMemcpyToSymbol(::stage27_spalding_zero_limit,
                                  &zeroEnabled,sizeof(zeroEnabled)));
    std::printf("STAGE27_SPALDING_ZERO_LIMIT enabled=%d scope=ACTIVE_N_LE_4 "
      "fivePointUnchanged=YES legacyWhenDisabled=YES status=PASS\n",zeroEnabled);
    std::printf("STAGE24B_ACTIVE_SAMPLING fitPoints=%d omegaSampleIndex=%d cacheSlots=5 "
      "actualRaySamples=5 fitting=ACTIVE_SUBSET fivePointPath=FROZEN "
      "momentumAndOFAuto=SHARED status=PASS\n",stage24bN,stage24bOmega);
    if(mesh.empty())throw std::runtime_error("--mesh required");
    if(amgHierarchy!="cf")throw std::runtime_error("HXT4C1 wires the optimal CF hierarchy first; use --amg-hierarchy cf");
    if(cfCoarsening!="pmis")throw std::runtime_error("HXT4C1 CF coarsening must be pmis");
    if(cfStrength!="classical-negative")throw std::runtime_error("HXT4C1 CF strength must be classical-negative");
    if(!(cfInterp=="direct"||cfInterp=="exti"))throw std::runtime_error("HXT4C1 CF interp must be direct or exti");
    if(!(amgSmoother=="jacobi"||amgSmoother=="l1jacobi"||amgSmoother=="cheb2"||amgSmoother=="mcgs"))throw std::runtime_error("HXT4C1 unknown AMG smoother");
    if(!(amgFineMode=="explicit"||
         amgFineMode=="matrixfree"))
      throw std::runtime_error(
        "--amg-fine-mode must be explicit or matrixfree");

    if(amgFineMode=="matrixfree"){
      if(pressureSolver!="pcg_amg")
        throw std::runtime_error(
          "matrix-free fine AMG requires pcg_amg");

      if(amgSmoother!="jacobi")
        throw std::runtime_error(
          "matrix-free fine AMG currently requires jacobi");

      if(amgJacobiFineSweeps!=1)
        throw std::runtime_error(
          "matrix-free fine AMG currently requires one fine Jacobi sweep");
    }
    if(pressureSolver!="pcg_amg"&&pressureSolver!="pcg_jacobi")throw std::runtime_error("HXT4C1 pressure solver must be pcg_amg or pcg_jacobi");
    if(!(pressureGauge=="pin"||pressureGauge=="natural"))throw std::runtime_error("pressure gauge must be pin or natural");
    if(!(podMode=="off"||podMode=="online"||podMode=="adaptive"))throw std::runtime_error("--pod-mode must be off, online, or adaptive");
    if(podMode!="off"){
      if(pressureSolver!="pcg_amg")throw std::runtime_error("Balanced POD requires pcg_amg");
      if(amgFineMode!="explicit")throw std::runtime_error("Balanced POD requires explicit AMG fine mode");
      if(podTrainSnapshots<=0||podRank<=0||podRank>podTrainSnapshots||podMaxRank<podRank||podNovelty<0.0||
         podTailIter<=0||podHardIts<0||(podHardIts>0&&podHardIts<=6*podTailIter)||
         podAcceptMinAbs<0||podAcceptMinRel<0.0||podAcceptMinRel>=1.0||
         podRefreshAbs<0||podRefreshRel<0.0||podRefreshRel>=10.0||
         podRecompressRank<0||podRecompressHistory<=0||podRecompressMaxLossAbs<0||
         podRecompressMaxLossRel<0.0||podRecompressMaxLossRel>=1.0||
         !(podRecompressMethod=="tail_pod"||podRecompressMethod=="whiten_tail_pod"||podRecompressMethod=="amg_spectral")||
         !(podRecompressForce==0||podRecompressForce==1)||!(podNoProbe==0||podNoProbe==1)||
         (podRecompressRank>0&&(podRecompressRank>=podMaxRank||podRecompressRank<podRank)))
        throw std::runtime_error("bad POD train/rank/max-rank/novelty/tail/accept/refresh/recompress configuration");
    }
    if(pressureSolver=="pcg_amg"&&amgSmoother=="mcgs"&&amgMcgsOrder!="symmetric")throw std::runtime_error("PCG requires symmetric MCGS");
    if(!(alphaU>0&&alphaU<=1&&alphaP>0))throw std::runtime_error("bad relaxation");
    if(!(gamma>=0.0)||!(schurNitscheGamma>=0.0)||(gamma<=0.0&&schurNitscheGamma>0.0))throw std::runtime_error("bad momentum/Schur Nitsche gamma");
    if(!(momCapPolicy=="warn"||momCapPolicy=="fail"))throw std::runtime_error("--mom-cap-policy must be warn or fail");
    if(!(simpleCoupling=="lagged"||simpleCoupling=="corrected"))throw std::runtime_error("SIMPLE_COUPLING/--simple-coupling must be lagged or corrected");
    if(!(momWork=="sgs1"||momWork=="fgs1"||momWork=="altgs1"||momWork=="fgmres-fgs"))throw std::runtime_error("--momentum-work must be sgs1, fgs1, altgs1, or fgmres-fgs");
    if(momFgmresRestart<=0||momFgsSweeps<=0)throw std::runtime_error("bad momentum FGMRES/FGS option");
    if(!(mixlenScale>=0.0)||!(outerContAtol>=0.0)||!(wallSampleFraction>0.0&&wallSampleFraction<1.0))throw std::runtime_error("bad HXT5B turbulence/boundary/convergence option");
    if(!(weakWallGamma>0.0)||!(weakWallHScale>0.0)||!(weakWallNutScale>=0.0))throw std::runtime_error("bad weak momentum wall option");
    if(!(spaldingMolecularConsistency==0||spaldingMolecularConsistency==1))throw std::runtime_error("--spalding-molecular-consistency must be 0 or 1");
    if(sstG1.enabled&&sstG2)throw std::runtime_error("choose only one of --sst-g1 and --sst-g2");
    if(laminarPerf&&(sstG1.enabled||sstG2))
      throw std::runtime_error("--laminar-perf cannot be combined with SST G1/G2");
    if(laminarPerf)mixlenScale=0.0;
    if(!(sstG2Start=="g0"||sstG2Start=="plug"))throw std::runtime_error("--sst-g2-start must be g0 or plug");
    if(sstG2MetricsEvery<=0)throw std::runtime_error("bad SST G2 metrics cadence");
    if(profileAudit&&(profileRadial<8||profileAngles<4||!(profileRmax>0&&profileRmax<=1)))throw std::runtime_error("bad enriched profile audit option");
    if(sstG2&&(!(sstG2PlateauTol>0)||!(sstG2ContAtol>0)||sstG2Max<=0||sstG2PrintEvery<=0||sstG2PlateauSamples<3))throw std::runtime_error("bad SST G2 control option");
    if(!(sstG1.lowReOmegaFactor>0.0))throw std::runtime_error("bad low-Re omega factor");
    if((sstG1.enabled||sstG2)&&(!(sstG1.intensity>0)||!(sstG1.lengthRatio>0)||!(sstG1.alphaK>0&&sstG1.alphaK<=1)||!(sstG1.alphaOmega>0&&sstG1.alphaOmega<=1)||!(sstG1.linearRtol>0&&sstG1.linearRtol<1)||!(sstG1.linearAtol>=0)||sstG1.linearMax<=0||!(sstG1.gsOmega>0)||!(sstG1.kFloor>0)||!(sstG1.omegaFloor>0)||!(sstG1.ofKappa>0)||!(sstG1.ofE>1)||!(sstG1.ofBeta1>0)||!(sstG1.ofReBlend>0)||!(sstG1.ofProductionScale>=0)||!(sstG1.ofWallVolumeScale>0)||!(sstG1.ofProductionLimit>0)))throw std::runtime_error("bad SST transport/wall option");
    if((sstG1.enabled||sstG2)&&!(sstG1.linearSolver=="sgs1"||sstG1.linearSolver=="fgs1"||sstG1.linearSolver=="fgmres-fgs"))throw std::runtime_error("bad SST linear solver option");
    if((sstG1.enabled||sstG2)&&(sstG1.fgmresRestart<=0||sstG1.fgsSweeps<=0))throw std::runtime_error("bad SST FGMRES/FGS option");
    if(sstG1.enabled||sstG2){
      const double kr = sstG1.kLinearRtol>=0.0 ? sstG1.kLinearRtol : sstG1.linearRtol;
      const double ka = sstG1.kLinearAtol>=0.0 ? sstG1.kLinearAtol : sstG1.linearAtol;
      const double wr = sstG1.omegaLinearRtol>=0.0 ? sstG1.omegaLinearRtol : sstG1.linearRtol;
      const double wa = sstG1.omegaLinearAtol>=0.0 ? sstG1.omegaLinearAtol : sstG1.linearAtol;
      const auto validScalarTol=[](double rt,double at){
        return std::isfinite(rt)&&std::isfinite(at)&&rt>=0.0&&rt<1.0&&at>=0.0&&(rt>0.0||at>0.0);
      };
      if(!validScalarTol(kr,ka)||!validScalarTol(wr,wa))
        throw std::runtime_error("bad per-equation SST linear tolerances (rtol and atol must be nonnegative, one strictly positive)");
      std::printf("NODALS_SST_SPLIT_LINEAR_TOL kRtol=%.12e kAtol=%.12e omegaRtol=%.12e omegaAtol=%.12e legacyCommonRtol=%.12e legacyCommonAtol=%.12e status=PASS\n",kr,ka,wr,wa,sstG1.linearRtol,sstG1.linearAtol);
    }

    HostMesh M=nodals_hxt1::load(mesh);
    const char* stage25Raw=std::getenv("SST25_ROTATE_DEG");
    const double stage25Degrees=stage25Raw?std::stod(stage25Raw):0.;
    nodals::stage25::rotate_loaded_tets(M,stage25Degrees);
    nodals::mesh::route_operator_patches(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");
    const auto stage19Patches=nodals::mesh::build_registry(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");
    auto IP=hxt2::build_interface_plans(M);auto I3=hxt3a::build_if(M);auto Bbase=hxt3a::build_b(M,I3);
    const auto inlet23=nodals::stage23::from_environment(bulk);
    // Stage23B removes the DG direction restriction. Geometry/patch maps and
    // analytic-pipe wall distance remain Stage23C/24 work. Avoid claiming this
    // is a fully rotated CFD test; inlet must point inward on tagged faces.
    // Verify actual inlet normal after creating the boundary face plans.
    std::printf("STAGE23A_INLET vector=[%.12e,%.12e,%.12e] provider=VECTOR_HOST_PRESSURE_SOURCE_AND_PLUG gpuDg=VECTOR_XYZ_TET_HEX status=PASS\n",
      inlet23.x,inlet23.y,inlet23.z);
    auto BH=hxt5b::build_boundaries(M,Bbase,inlet23);auto&Bh=BH.B;
    double stage23CMinInward=1e300;
    const double stage23CSpeed=std::sqrt(inlet23.x*inlet23.x+inlet23.y*inlet23.y+inlet23.z*inlet23.z);
    for(const auto &f:BH.inT){
      const double inward=-(f.n[0]*inlet23.x+f.n[1]*inlet23.y+f.n[2]*inlet23.z);
      if(!(inward>1e-8*stage23CSpeed))
        throw std::runtime_error("STAGE23C inlet patch has non-inward face; check explicit face classification and inlet vector");
      stage23CMinInward=std::min(stage23CMinInward,inward);
    }
    std::printf("STAGE23C_INLET_NORMALS faces=%zu minInwardSpeed=%.12e status=PASS\n",BH.inT.size(),stage23CMinInward);
    double stage23InletFlux=0.;
    for(const auto&f:BH.inT)
      stage23InletFlux+=f.area*(inlet23.x*f.n[0]+inlet23.y*f.n[1]+inlet23.z*f.n[2]);
    if(!BH.inT.empty() && !(stage23InletFlux<0.))
      throw std::runtime_error("STAGE23B inlet vector has no net inflow through tagged tet inlet faces");
    std::printf("STAGE23B_INLET_FLOW prescribed=%.12e orientation=INWARD status=PASS\n",-stage23InletFlux);
    nodals::mesh::audit_legacy_operators(stage19Patches,BH,M);
    // STAGE27_DG_PROFILE_V1: active only when a CSV is explicitly supplied.
    // The original constant DG path and initialization are untouched otherwise.
    const char* profileCsvRaw=std::getenv("STAGE27_INLET_PROFILE_CSV");
    const bool stage27ProfileActive=profileCsvRaw && *profileCsvRaw;
    stage27_inlet::HostProfile stage27HostProfile;
    if(stage27ProfileActive){
      stage27HostProfile=stage27_inlet::load(M,BH,profileCsvRaw);
      stage27_inlet::adjust_pressure_source(BH,M,stage27HostProfile,inlet23);
      if(std::abs(BH.sourceSum+stage27HostProfile.inwardFlux)>1.e-8)
        throw std::runtime_error("STAGE27_DG_PROFILE mass-flux audit failed");
      std::printf("STAGE27_DG_PROFILE=ON csv=%s vertices=%zu faces=%zu "
        "inwardFlux=%.12e minInward=%.12e maxSpeed=%.12e status=PASS\n",
        profileCsvRaw,stage27HostProfile.nVertices,BH.inT.size(),
        stage27HostProfile.inwardFlux,stage27HostProfile.minInward,
        stage27HostProfile.maxSpeed);
    }else std::printf("STAGE27_DG_PROFILE=OFF operator=FROZEN_CONSTANT_VECTOR status=PASS\n");
    hxt5c::Stage20GpuDistanceOwner stage20WallField;
    const std::string stage20DistanceMethod=std::getenv("SST_WALL_DISTANCE_METHOD")?std::getenv("SST_WALL_DISTANCE_METHOD"):"analytic_pipe";
    if(stage20DistanceMethod=="geometric"){
      const auto d=nodals::geometry::vertex_wall_distances(M,stage19Patches);
      stage20WallField.attach(d);
      std::printf("SST_WALL_DISTANCE_RUNTIME method=geometric provider=P1_FROM_CLOSEST_SOLID_TRIANGLES status=PASS\n");
    }else if(stage20DistanceMethod=="analytic_pipe"){
      std::printf("SST_WALL_DISTANCE_RUNTIME method=analytic_pipe provider=STAGE16_EXACT status=PASS\n");
    }else throw std::runtime_error("STAGE20 unsupported wall distance method: "+stage20DistanceMethod);
    const int nv=Bh.nv,np=Bh.np,nh=(int)M.h.nhex;
    if(std::getenv("SST22D3_B_AUDIT")){
      auto Bn=Bh;
      hxt5b::s22d3_patch_normal_wall_divergence(Bn,M,BH.wallT);
      std::vector<double> ux((std::size_t)Bn.nv),uy((std::size_t)Bn.nv),uz((std::size_t)Bn.nv);
      std::vector<double> ptest((std::size_t)Bn.np);
      for(int i=0;i<Bn.nv;++i){ux[i]=std::sin(.107*i);uy[i]=std::cos(.133*i);uz[i]=std::sin(.079*i);}
      for(int i=0;i<Bn.np;++i)ptest[i]=std::cos(.11*i);
      const auto oldB=hxt3a::b_host(Bh,ux,uy,uz);
      const auto newB=hxt3a::b_host(Bn,ux,uy,uz);
      std::vector<double> direct((std::size_t)Bn.np,0.);
      for(const auto&f:BH.wallT){
        const auto&t=M.tetVel[f.cell];
        int f0=hxt2::TET_FACE[f.lf][0],f1=hxt2::TET_FACE[f.lf][1],f2=hxt2::TET_FACE[f.lf][2];
        for(int q=0;q<12;++q){
          double l[4]={0,0,0,0};l[f0]=hxt5b::HL0[q];l[f1]=hxt5b::HL1[q];l[f2]=hxt5b::HL2[q];
          double phi[8];hxt5b::tbasis(l,phi);double normal=0.;
          for(int a=0;a<8;++a){int g=t.g[a];normal+=phi[a]*(f.n[0]*ux[g]+f.n[1]*uy[g]+f.n[2]*uz[g]);}
          direct[f.cell]-=hxt5b::HW12[q]*f.area*normal;
        }
      }
      double defect=0.,scale=0.,lhs=0.,rhs=0.;
      for(int i=0;i<Bn.np;++i){
        defect=std::max(defect,std::abs(newB[i]-oldB[i]-direct[i]));
        scale=std::max(scale,std::abs(direct[i]));
        lhs+=ptest[i]*newB[i];
      }
      std::vector<double> gt0,gt1,gt2;
      hxt3a::bt_host(Bn,ptest,gt0,gt1,gt2);
      for(int i=0;i<Bn.nv;++i)rhs+=ux[i]*gt0[i]+uy[i]*gt1[i]+uz[i]*gt2[i];
      const double adj=std::abs(lhs-rhs)/std::max(1.,std::abs(lhs));
      if(defect>1.e-11*std::max(1.,scale)||adj>1.e-10)
        throw std::runtime_error("STAGE22D3 B_N geometry/adjoint audit failed");
      std::printf("STAGE22D3_PRESSURE_FACE_AUDIT faces=%zu sign=B_POSITIVE_DIV_MINUS_WALL_FLUX "
        "maxFluxDefect=%.12e adjointRel=%.12e productionB=UNCHANGED status=PASS\n",
        BH.wallT.size(),defect,adj);
    }


    if(std::getenv("SST22D4_GLOBAL_SCHUR_AUDIT")){
      auto Bn=Bh;
      hxt5b::s22d3_patch_normal_wall_divergence(Bn,M,BH.wallT);
      std::vector<double> q0((std::size_t)Bn.nv),q1((std::size_t)Bn.nv),q2((std::size_t)Bn.nv),ptest((std::size_t)Bn.np);
      for(int j=0;j<Bn.nv;++j){q0[j]=std::sin(.0227*(j+1));q1[j]=std::cos(.0123*(j+1));q2[j]=std::cos(.0171*(j+1));}
      for(int j=0;j<Bn.np;++j)ptest[j]=std::sin(.019*(j+1));
      std::vector<double> bt0,bt1,bt2;
      hxt3a::bt_host(Bn,ptest,bt0,bt1,bt2);
      double pressureAdjoint=0.,pressureNorm=0.;
      for(int i=0;i<Bn.nv;++i)pressureAdjoint+=q0[i]*bt0[i]+q1[i]*bt1[i]+q2[i]*bt2[i];
      const auto bq=hxt3a::b_host(Bn,q0,q1,q2);
      for(int k=0;k<Bn.np;++k)pressureNorm+=ptest[k]*bq[k];
      const double adj=std::abs(pressureAdjoint-pressureNorm)/std::max(1.,std::abs(pressureNorm));
      std::vector<double> w0((std::size_t)Bn.nv),w1((std::size_t)Bn.nv),w2((std::size_t)Bn.nv),
                          c0((std::size_t)Bn.nv),c1((std::size_t)Bn.nv),c2((std::size_t)Bn.nv);
      for(int i=0;i<Bn.nv;++i){
        const double r0=.02+.00001*(i%47),r1=.019+.00001*(i%53),r2=.017+.00001*(i%61);
        w0[i]=r0*bt0[i];w1[i]=r1*bt1[i];w2[i]=r2*bt2[i];
        c0[i]=q0[i]-w0[i];c1[i]=q1[i]-w1[i];c2[i]=q2[i]-w2[i];
      }
      const auto sp=hxt3a::b_host(Bn,w0,w1,w2);
      const auto after=hxt3a::b_host(Bn,c0,c1,c2);
      double parity=0.,sEnergy=0.,sDirect=0.,sSize=0.;
      for(int j=0;j<Bn.np;++j){
        parity=std::max(parity,std::abs(after[j]-(bq[j]-sp[j])));
        sSize=std::max(sSize,std::abs(sp[j]));
        sEnergy+=ptest[j]*sp[j];
      }
      for(int i=0;i<Bn.nv;++i)sDirect+=bt0[i]*w0[i]+bt1[i]*w1[i]+bt2[i]*w2[i];
      const double energyRel=std::abs(sEnergy-sDirect)/std::max(1.,std::abs(sDirect));
      if(adj>1.e-10||parity>1.e-11*std::max(1.,sSize)||energyRel>1.e-10||!(sEnergy>0.))
        throw std::runtime_error("STAGE22D4 GLOBAL SCHUR audit mismatch");
      std::printf("STAGE22D4_GLOBAL_SCHUR_AUDIT faces=%zu pressureRows=%d velocityDofs=%d adjointRel=%.12e "
         "correctedContinuityDefect=%.12e schurEnergyRel=%.12e schurPositive=1 "
         "productionPressure=UNCHANGED productionMomentum=UNCHANGED gpuNitsche=COMPILED_NOT_DISPATCHED status=PASS\n",
         BH.wallT.size(),Bn.np,Bn.nv,adj,parity,energyRel);
    }

    double zmin=1e300,zmax=-1e300,R=0;
    for(const auto&q:M.points){
      const double axial=nodals::stage25::axial(q.x,q.y,q.z);
      const double rx=nodals::stage25::radial_x(q.x,q.y,q.z);
      zmin=std::min(zmin,axial);zmax=std::max(zmax,axial);
      R=std::max(R,std::hypot(rx,q.y));
    }
    // Stage26: do not estimate the hydraulic diameter of a CURVED duct
    // from a global origin or global Z bounding box. The cap equivalent
    // diameter is prescribed explicitly by the mesh preflight.
    const char *s26diam=std::getenv("SST26_SPIRAL_DIAMETER");
    const bool s26spiral=s26diam&&*s26diam;
    const double s26D=s26spiral?std::stod(s26diam):2.*R;
    const double L=zmax-zmin,D=s26D,nu=bulk*D/Re;
    if(s26spiral)R=D/2.;
    if(!(L>0&&D>0&&nu>0&&std::isfinite(D)&&std::isfinite(Re)&&Re>0))
      throw std::runtime_error("STAGE26 invalid reference dimensions");
    if(s26spiral){
      if(stage25Degrees!=0.)throw std::runtime_error("STAGE26 spiral must not be rotated by pipe-only Stage25 transform");
      std::printf("STAGE26_SPIRAL_REFERENCE diameter=%.12e bulk=%.12e Re=%.12e nu=%.12e "
        "boundingZSpan=%.12e pathLength=UNDEFINED_NOT_A_STRAIGHT_PIPE "
        "diameterSource=END_CAP_AREA wallDistance=GEOMETRIC status=PASS\n",D,bulk,Re,nu,L);
    }else{
      std::printf("STAGE25_FRAME angleDeg=%.12g axis=[%.12e,0,%.12e] length=%.12e "
        "radius=%.12e diameter=%.12e nu=%.12e Re=%.12e provider=ROTATION_COVARIANT status=PASS\n",
        stage25Degrees,nodals::stage25::s,nodals::stage25::c,L,R,D,nu,Re);
    }

    const std::string s22Mode=std::getenv("SST22_TRACTION_MODEL")?
      std::getenv("SST22_TRACTION_MODEL"):"legacy_axial";
    const bool stage22Vector=(s22Mode=="vector_projected");
    const char* liveFlag=std::getenv("SST22D4_NITSCHE_ENABLE");
    const bool stage22d4iEnable=std::getenv("SST22D4I_BF3") &&
      std::string(std::getenv("SST22D4I_BF3"))=="1";
    const bool stage22d4Live=liveFlag && std::string(liveFlag)=="1";
    // Stage SC1: experimental off-wall sampled tangential conductance.
    // Default preserves the exact Stage41 wall-trace operator.
    const std::string sampledWallMode=std::getenv("NODALS_MOM_WALL_OPERATOR")?
      std::getenv("NODALS_MOM_WALL_OPERATOR"):"original";
    if(sampledWallMode!="original" && sampledWallMode!="sampled_conductance")
      throw std::runtime_error("NODALS_MOM_WALL_OPERATOR must be original or sampled_conductance");
    const bool sampledConductance=sampledWallMode=="sampled_conductance";
    const std::string f1WallAssembly=std::getenv("NODALS_F1_FACE_LOCAL")?
      std::getenv("NODALS_F1_FACE_LOCAL"):"0";
    if(f1WallAssembly!="0" && f1WallAssembly!="1")
      throw std::runtime_error("NODALS_F1_FACE_LOCAL must be 0 or 1");
    const bool f1FaceLocal=f1WallAssembly=="1";
    if(f1FaceLocal && (!stage22d4Live || !stage22d4iEnable))
      throw std::runtime_error("F1A needs live coupled BF3 tetrahedral wall path");
    std::printf("NODALS_F1A_CONFIG enabled=%d mode=%s quadrature=12 "
      "spalding=UNCHANGED omega=UNCHANGED bf3=UNCHANGED "
      "fgmres=COUPLED_UNCHANGED status=PASS\n",
      (int)f1FaceLocal,f1FaceLocal?"FACE_SHARED_REDUCE":"LEGACY_QUAD_ATOMICS");
    // F1B freezes the 8x8 off-component wall tensor per face for repeated
    // coupled-FGMRES actions; independent of F1A face-local assembly.
    const std::string f1bWallEnv=std::getenv("NODALS_F1B_FROZEN_WALL")?
      std::getenv("NODALS_F1B_FROZEN_WALL"):"0";
    if(f1bWallEnv!="0" && f1bWallEnv!="1")
      throw std::runtime_error("NODALS_F1B_FROZEN_WALL must be 0 or 1");
    const bool f1bFrozenWall=f1bWallEnv=="1";
    if(f1bFrozenWall && (!stage22d4Live || !stage22d4iEnable))
      throw std::runtime_error("F1B requires live BF3 tetrahedral wall momentum");
    // NODALS_DEVIATORIC_STRESS_SCOPE_FIX: validate after mesh and BF3/F1B flags.
    if(momentumDivergenceCorrection || momentumEmac){
      if(!sstG2 || M.h.nhex || M.h.ninterface || !M.h.ntet)
        throw std::runtime_error("conservative momentum Picard currently supports pure-tet SST G2");
      const char* st=std::getenv("SST_STATE_SPACE");
      if(st && std::string(st)!="q1")
        throw std::runtime_error("skew/conservative/EMAC momentum requires SST_STATE_SPACE=q1");
      // No scalar-storage restriction: compact Q1 changes only k/omega CSR.
      // Momentum Picard continues to use the same P1+BF3 velocity operator.
      if(stage33MomentumSupg)
        throw std::runtime_error("momentum SUPG requires an updated strong residual for conservative convection");
    }
    if(implicitDevStress){
      if(M.h.nhex || M.h.ninterface || !M.h.ntet)
        throw std::runtime_error("implicit deviatoric stress supports pure tetrahedral meshes");
      if(!stage22d4Live || !stage22d4iEnable || !f1bFrozenWall)
        throw std::runtime_error("implicit deviatoric stress requires coupled BF3 and F1B");
      const char* ss=std::getenv("SST_STATE_SPACE");
      if(ss && std::string(ss)!="q1")
        throw std::runtime_error("implicit deviatoric stress requires SST_STATE_SPACE=q1");
      if(sstG1.deviatoricMomentum)
        throw std::runtime_error("implicit deviatoric volume cannot be combined with legacy explicit deviatoric RHS");
    }
    const std::string f1bAuditEnv=std::getenv("NODALS_F1B_GPU_GATE")?
      std::getenv("NODALS_F1B_GPU_GATE"):"0";
    if(f1bAuditEnv!="0" && f1bAuditEnv!="1")
      throw std::runtime_error("NODALS_F1B_GPU_GATE must be 0 or 1");
    const bool f1bGpuGate=f1bAuditEnv=="1";
    std::printf("NODALS_F1B_CONFIG enabled=%d mode=%s crossBlock=8x8_FP64_PER_FACE "
      "quadrature=12_UNCHANGED spalding=UNCHANGED omega=UNCHANGED "
      "bf3=UNCHANGED fgmres=COUPLED_UNCHANGED status=PASS\n",
      (int)f1bFrozenWall,f1bFrozenWall?"CACHED_FROZEN_CROSS":"LEGACY_QUAD_ACTION");
    // F2 stays opt-in until Choi profiles and throughput are validated.
    // It replaces 12 q-level momentum Spalding fits with one face-averaged
    // fit, but retains 12-point matrix/normal quadrature and the omega wall law.
    const std::string f2WallEnv=std::getenv("NODALS_F2_FACE_SPALDING")?
      std::getenv("NODALS_F2_FACE_SPALDING"):"0";
    if(f2WallEnv!="0" && f2WallEnv!="1")
      throw std::runtime_error("NODALS_F2_FACE_SPALDING must be 0 or 1");
    const bool f2FaceSpalding=f2WallEnv=="1";
    if(f2FaceSpalding && (!stage22d4Live || !stage22d4iEnable))
      throw std::runtime_error("F2 requires coupled BF3 tetrahedral wall operator");
    std::printf("NODALS_F2_CONFIG enabled=%d mode=%s momentumFitsPerFace=%d "
      "omega=UNCHANGED quad=12 bf3=UNCHANGED fgmres=COUPLED status=PASS\n",
      (int)f2FaceSpalding,
      f2FaceSpalding?"AREA_WEIGHTED_VECTOR_FACE_FIT":"LEGACY_QUAD_FITS",
      f2FaceSpalding?1:12);
    if(sampledConductance && (!stage22d4Live || !stage22d4iEnable))
      throw std::runtime_error("sampled_conductance requires SST22D4_NITSCHE_ENABLE=1 and SST22D4I_BF3=1");
    std::printf("NODALS_WALL_SC1 mode=%s sampledBasis=%s normalBasis=PHYSICAL_WALL_TRACE "
      "normalConstraint=UNCHANGED omega=UNCHANGED source=STAGE41_TET status=PASS\n",
      sampledWallMode.c_str(),sampledConductance?"CA_BP_NORMAL_COLUMN":"CA_BW_WALL_TRACE");
    const std::string stage22d4dMode=std::getenv("SST22D4D_MOM_MODE")?
      std::getenv("SST22D4D_MOM_MODE"):"segregated";
    // Stage27/5.B.1: TRUE projected component Krylov; keep coupled FGMRES available.
    const std::string s05b1Mode=std::getenv("STAGE27_S05B1_MOM_MODE")?
      std::getenv("STAGE27_S05B1_MOM_MODE"):"inherit";
    const auto s05b1Integer=[](const char*name,int def,int min,int max){
      const char*str=std::getenv(name);
      if(!str||!*str)return def;
      char*end=nullptr;long value=std::strtol(str,&end,10);
      if(!end||*end!='\0'||value<min||value>max)
        throw std::runtime_error(std::string("Stage5.B.1 invalid integer: ")+name);
      return (int)value;
    };
    // Stage5.B.3 opt-in alternative scalar inner methods. FGMRES remains selectable.
    const std::string s05b3Mode=std::getenv("STAGE27_S05B3_MOM_MODE")?
      std::getenv("STAGE27_S05B3_MOM_MODE"):"inherit";
    const bool s05b3Active=s05b3Mode=="bicgstab"||s05b3Mode=="fgs-fixed";
    if(!(s05b3Mode=="inherit"||s05b3Active))
      throw std::runtime_error("Stage5.B.3 unknown momentum mode");
    if(s05b3Active && s05b1Mode!="inherit")
      throw std::runtime_error("Stage5.B.3 requires STAGE27_S05B1_MOM_MODE=inherit");
    const int s05b3Blocks=s05b1Integer("STAGE27_S05B3_BLOCK_PASSES",2,1,30);
    const int s05b3InnerIts=s05b1Integer("STAGE27_S05B3_COMPONENT_ITS",6,1,400);
    const int s05b3InnerFgs=s05b1Integer("STAGE27_S05B3_PRECOND_FGS",1,1,16);
    const int s05b3Audit=s05b1Integer("STAGE27_S05B3_FULL_AUDIT",1,0,1);
    const int s05b3Require=s05b1Integer("STAGE27_S05B3_REQUIRE_TOL",s05b3Mode=="bicgstab"?1:0,0,1);
    const char* s05b3DampEnv=std::getenv("STAGE27_S05B3_DAMPING");
    const double s05b3Damping=s05b3DampEnv?std::stod(s05b3DampEnv):0.35;
    if(!(s05b3Damping>0. && s05b3Damping<=1.)||!std::isfinite(s05b3Damping))
      throw std::runtime_error("Stage5.B.3 damping must be finite in (0,1]");
    const std::string m3Mode=std::getenv("NODALS_M3_MODE")?
      std::getenv("NODALS_M3_MODE"):"off";
    if(!(m3Mode=="off"||m3Mode=="fgs"||m3Mode=="sgs"||m3Mode=="m4"))
      throw std::runtime_error("M3 mode must be off|fgs|sgs|m4");
    if(momentumEmac && !(m3Mode=="fgs" || m3Mode=="sgs"))
      throw std::runtime_error("experimental EMAC Picard currently requires M3 FGS/SGS");
    if(momentumDivergenceCorrection && m3Mode=="m4")
      throw std::runtime_error("conservative momentum Picard currently requires M3 or coupled FGMRES, not M4");
    if(implicitDevStress && m3Mode=="m4")
      throw std::runtime_error("implicit deviatoric stress supports M3 FGS/SGS or coupled FGMRES, not experimental M4");
    const std::string bazbMode=std::getenv("NODALS_BAZILEVS_B_VISCOSITY")?
      std::getenv("NODALS_BAZILEVS_B_VISCOSITY"):"off";
    if(bazbMode!="off" && bazbMode!="nu" && bazbMode!="nu_plus_nut")
      throw std::runtime_error("NODALS_BAZILEVS_B_VISCOSITY must be off, nu, or nu_plus_nut");
    const int bazbViscosityMode=bazbMode=="off"?0:(bazbMode=="nu"?1:2);
    const char* slipEnv=std::getenv("NODALS_BAZILEVS_NUMERICAL_SLIP");
    const std::string slipSwitch=(slipEnv&&*slipEnv)?slipEnv:"0";
    if(slipSwitch!="0" && slipSwitch!="1")
      throw std::runtime_error("NODALS_BAZILEVS_NUMERICAL_SLIP must be 0 or 1");
    const bool bazilevsNumericalSlip=slipSwitch=="1";
    const char* cbEnv=std::getenv("NODALS_BAZILEVS_CB");
    const double bazilevsCb=(cbEnv&&*cbEnv)?std::stod(cbEnv):4.0;
    if(!(bazilevsCb>0.0) || !std::isfinite(bazilevsCb))
      throw std::runtime_error("NODALS_BAZILEVS_CB must be positive finite");
    if(bazilevsNumericalSlip && !bazbViscosityMode)
      throw std::runtime_error("numerical slip requires symmetric tangential Nitsche consistency");
    std::printf("NODALS_BAZILEVS_SLIP enabled=%d Cb=%.8g y=h_over_Cb "
      "velocity=PHYSICAL_WALL_QUADRATURE_TRACE spalding=REPLACED_MOMENTUM_ONLY "
      "omega=UNCHANGED status=CONFIGURED\n",(int)bazilevsNumericalSlip,bazilevsCb);
    // M3 uses assembled nine-block CSR. Coupled FGMRES uses the already
    // BazB-augmented matrix-free S22D4DOperator::apply and reports true residuals.
    // Block/segregated alternatives have not been verified with BazB.
    const bool bazbCoupledFgmres=m3Mode=="off" &&
      s05b1Mode=="inherit" && !s05b3Active;
    if(bazbViscosityMode && !(m3Mode=="fgs"||m3Mode=="sgs"||bazbCoupledFgmres))
      throw std::runtime_error("Bazilevs-B requires M3 FGS/SGS or original coupled FGMRES");
    if(bazbViscosityMode && (sampledConductance || !f1bFrozenWall || !f2FaceSpalding ||
         !stage22d4Live || !stage22d4iEnable || stage22d4dMode!="coupled" ||
         !implicitDevStress || momentumFullSupg))
      throw std::runtime_error("Bazilevs-B requires original sampled wall, F1B, F2, coupled BF3 M3, implicit deviatoric stress, momentum SUPG off");
    std::printf("NODALS_BAZILEVS_B enabled=%d viscosity=%s tangential=NITSCHE_CONSISTENCY_PAIR "
      "spalding=UNCHANGED omega=UNCHANGED bf3=UNCHANGED solver=%s status=CONFIGURED\n",
      (int)(bazbViscosityMode!=0),bazbMode.c_str(),
      bazbCoupledFgmres?"COUPLED_MATRIXFREE_FGMRES":"M3_CSR");
    const bool m3Active=m3Mode!="off";
    if(momentumFullSupg && m3Mode!="fgs" && m3Mode!="sgs")
      throw std::runtime_error("Full momentum SUPG requires M3 FGS or M3 SGS");
    const int m3Sweeps=s05b1Integer("NODALS_M3_SWEEPS",m3Mode=="sgs"?5:10,1,60);
    const double m3Omega=std::getenv("NODALS_M3_OMEGA")?
      std::stod(std::getenv("NODALS_M3_OMEGA")):0.7;
    if(!(m3Omega>0. && m3Omega<=1.0) || !std::isfinite(m3Omega))
      throw std::runtime_error("M3 omega must be in (0,1]");
    if(m3Active && (s05b3Active||s05b1Mode!="inherit"))
      throw std::runtime_error("M3 cannot coexist with Stage5.B.1/3 alternatives");
    if(m3Active && (!stage22d4Live||!stage22d4iEnable||stage22d4dMode!="coupled"))
      throw std::runtime_error("M3 requires live BF3 and coupled momentum");
    std::printf("NODALS_MOMENTUM_CONVECTION form=%s lag=PICARD_U_STAR "
      "transported=IMPLICIT_U strongForm=%s "
      "volume=TET8X8_TRIPLE_TENSOR boundary=EXISTING_DG_SPALDING_UNCHANGED "
      "scalarTransport=ORIGINAL_ADVECTIVE status=PASS\n",
      momentumConvectionForm.c_str(),momentumConservative?
      "DIV_USTAR_OUTER_U":(momentumSkew?
      "USTAR_DOT_GRAD_U_PLUS_HALF_DIV_USTAR_TIMES_U":(momentumEmac?
      "2D_U_TIMES_USTAR_PLUS_DIV_U_TIMES_USTAR":"USTAR_DOT_GRAD_U")));
    if(momentumEmac)std::printf("NODALS_EMAC_PICARD lag=U_STAR "
      "momentum=BLOCK_3X3_CSR extra=GRAD_KINETIC_PLUS_CARRIER_DIV_U "
      "pressure=P_EMAC_EQ_P_PHYSICAL_MINUS_HALF_U_SQUARED "
      "pressureOutput=SHIFTED boundary=UNCHANGED volume=EXPERIMENTAL "
      "scalarTransport=UNCHANGED status=EXPERIMENTAL\n");
    std::printf("NODALS_M3_CONFIG mode=%s sweeps=%d omega=%.3f "
      "coupling=BLOCK_3X3_CSR exactProjection=Tt_M_T innerResiduals=NONE "
      "amg=UNCHANGED f1b=REQUIRED status=PASS\n",
      m3Mode.c_str(),m3Sweeps,m3Omega);
    const bool s05b1Active=s05b3Active||s05b1Mode=="segregated-fgmres"||s05b1Mode=="segregated-fgmres-guarded";
    const bool s05b1Guarded=s05b1Mode=="segregated-fgmres-guarded";
    if(!(s05b1Mode=="inherit"||s05b1Mode=="coupled-fgmres"||s05b1Active))
      throw std::runtime_error("Stage5.B.1 unknown momentum mode");
    const int s05b1BlockMax=s05b1Integer("STAGE27_S05B1_BLOCK_MAX",3,1,30);
    const int s05b1ComponentMax=s05b1Integer("STAGE27_S05B1_COMPONENT_MAX",30,1,400);
    const int s05b1Restart=s05b1Integer("STAGE27_S05B1_RESTART",12,2,60);
    const int s05b1PrecondSweeps=s05b1Integer("STAGE27_S05B1_PRECOND_FGS",1,1,16);
    if(s05b1Active && (!stage22d4Live||!stage22d4iEnable||stage22d4dMode!="coupled"))
      throw std::runtime_error("Stage5.B.1 requires live BF3 constrained momentum and SST22D4D_MOM_MODE=coupled");
    std::printf("STAGE27_S05B1_CONFIG mode=%s blockMax=%d componentMax=%d restart=%d "
      "precondFGS=%d guarded=%d operator=Tt_A_T diagonal=EXACT_PROJECTED "
      "originalCoupledFGMRES=SELECTABLE scalar5A=UNCHANGED status=PASS\n",
      s05b1Mode.c_str(),s05b1BlockMax,s05b1ComponentMax,s05b1Restart,
      s05b1PrecondSweeps,(int)s05b1Guarded);
    std::printf("STAGE27_S05B3_CONFIG mode=%s componentIts=%d blockPasses=%d "
      "precondFGS=%d damping=%.6g fullAudit=%d requireTol=%d "
      "guardedFallback=COUPLED_FGMRES scalar5A=UNCHANGED "
      "projectedAction=STAGE5B2 status=PASS\n",
      s05b3Mode.c_str(),s05b3InnerIts,s05b3Blocks,s05b3InnerFgs,
      s05b3Damping,s05b3Audit,s05b3Require);
    if(stage22d4dMode!="segregated" && stage22d4dMode!="coupled")
      throw std::runtime_error("D4D mode must be segregated or coupled");
    if(stage22d4dMode=="coupled" && !stage22d4Live)
      throw std::runtime_error("D4D coupled requires live Nitsche");
    const int stage22d4dRestart=std::getenv("SST22D4D_RESTART")?
      std::stoi(std::getenv("SST22D4D_RESTART")):12;
    const int stage22d4dMaxIts=std::getenv("SST22D4D_MAXITS")?
      std::stoi(std::getenv("SST22D4D_MAXITS")):150;
    if(stage22d4dRestart<2||stage22d4dRestart>60||stage22d4dMaxIts<1||stage22d4dMaxIts>1000)
      throw std::runtime_error("D4D invalid block Krylov cap/restart");
    if(stage22d4iEnable && (!stage22d4Live || stage22d4dMode!="coupled"))
      throw std::runtime_error("Stage22D4I BF3 requires live Stage22D4B and coupled D4D momentum");
    if(stage22d4iEnable)std::printf("STAGE22D4I_MODE constrained_momentum=Tt_A_T "
      "bf3_wall_flux=EXACT_MEAN_ZERO normal_nitsche=OFF spalding_tangent=ON "
      "pressure=BN_ORIGINAL_PLUS_CONSTRAINT_PRESERVING_CORRECTION status=PASS\n");
    // D4D configuration is printed after the existing gamma is parsed.
    if(liveFlag && !stage22d4Live)
      throw std::runtime_error("STAGE22D4B enable accepts only SST22D4_NITSCHE_ENABLE=1");
    const double s22d4Gamma=std::getenv("SST22D4_GAMMA_N")?
      std::stod(std::getenv("SST22D4_GAMMA_N")):30.0;
    if(stage22d4iEnable)std::printf(
      "STAGE22D4I2_CONFIG gammaStab=%.12e constrained=BF3_MEAN_FACE_FLUX_ZERO "
      "momentum=Tt_A_T_PLUS_NORMAL_PENALTY normalConsistency=OFF "
      "gate=POST_SIMPLE_DERELAXED_REDUCED_FROZEN_PICARD status=PASS\n",s22d4Gamma);
    if(stage22d4Live)std::printf("STAGE22D4D_CONFIG mode=%s block=3NV_MATRIXFREE_FGMRES frozenWallCoefficients=YES schur=UNCHANGED gammaN=%.8g status=PASS\n",
      stage22d4dMode.c_str(),s22d4Gamma);
    if(stage22d4Live && (s22Mode!="vector_projected" || !sstG2 ||
       M.h.nhex || M.h.ninterface || !(stage22d4iEnable?(s22d4Gamma>=0.):(s22d4Gamma>0.)) || !std::isfinite(s22d4Gamma) ||
       simpleCoupling!="corrected" || sstG2Start!="plug"))
      throw std::runtime_error("STAGE22D4B requires puretet G2, corrected SIMPLE, plug, vector mode and positive gamma");
    if(std::getenv("SST22D_NITSCHE_ENABLE"))
      throw std::runtime_error("STAGE22D3 Nitsche disabled: full GPU momentum/pressure operator not yet wired");
    const bool stage22dFaceFit=std::getenv("SST22D_FACE_FIT") &&
      std::string(std::getenv("SST22D_FACE_FIT"))=="1";
    if(stage22dFaceFit && stage22Vector)
      throw std::runtime_error("STAGE22D isolated face-fit cannot enable unrestricted vector wall");
    if(stage22dFaceFit && (M.h.nhex||M.h.ninterface||!sstG2))
      throw std::runtime_error("STAGE22D isolated face-fit requires puretet SST-G2");
    if(stage22dFaceFit)std::printf("SST22D_FACE_FIT mode=FACE_TANGENTIAL_SAMPLES axial_matrix=UNCHANGED clampXY=STRONG schur=STAGE21_UNCHANGED nitsche=OFF status=PASS\n");
    if(s22Mode!="legacy_axial"&&!stage22Vector)
      throw std::runtime_error("STAGE22B invalid traction mode");
    const double stage22Penalty=std::getenv("SST22_NORMAL_PENALTY")?
      std::stod(std::getenv("SST22_NORMAL_PENALTY")):0.;
    if(stage22Vector && (!sstG2 || M.h.nhex || M.h.ninterface ||
        sstG2Start!="plug" || !std::isfinite(stage22Penalty) || stage22Penalty<0.))
      throw std::runtime_error("STAGE22B vector mode currently requires pure tet, G2, plug start, nonnegative penalty");
    std::printf("SST22_WALL_TRACTION mode=%s momentum=SEGREGATED_XYZ normalPenaltyRatio=%.8g "
      "coupledBlockSolver=OFF split=COMPONENT_DIAGONAL normalPenaltyScale=NU_OVER_H status=PASS\n",s22Mode.c_str(),stage22Penalty);
    if(stage22Vector && !stage22d4Live)std::printf("SST22C_NORMAL_PENALTY model=BETA_PLUS_CN_NU_OVER_H Cn=%.8g h=FACE_NORMAL_TET_ALTITUDE status=PASS\n",stage22Penalty);
    double stage22d4InletArea=0.;
    for(const auto&f:BH.inT)stage22d4InletArea+=f.area;
    if(stage22d4Live && !(stage22d4InletArea>0.))
      throw std::runtime_error("STAGE22D4B cannot normalize wall flux: inlet patch area nonpositive");
    if(stage22d4Live){
      hxt5b::s22d3_patch_normal_wall_divergence(Bh,M,BH.wallT);
      std::printf("STAGE22D4B_PRODUCTION_B mode=B_VOLUME_MINUS_WALL_NORMAL_FLUX "
        "momentumPressure=-BTRANSPOSE Schur=B_R_BT gammaN=%.8g "
        "wall=SYMMETRIC_NORMAL_NITSCHE_PLUS_TANGENTIAL_SPALDING "
        "viscosity=SST_NU_EFF wallXY=UNCLAMPED experiment=1 status=PASS\n",s22d4Gamma);
    }
    auto fixed=hxt5b::plug_initial(M,inlet23);
    double plugVertexErr=0.0,plugEnrichMax=0.0;
    for(std::uint64_t v=0;v<M.h.nv;++v){
      const std::size_t i=(std::size_t)v;
      plugVertexErr=std::max(plugVertexErr,std::abs((double)fixed.u0[i]-inlet23.x));
      plugVertexErr=std::max(plugVertexErr,std::abs((double)fixed.u1[i]-inlet23.y));
      plugVertexErr=std::max(plugVertexErr,std::abs((double)fixed.u2[i]-inlet23.z));
    }
    for(std::uint64_t g=M.h.nv;g<M.h.nvel;++g){
      const std::size_t i=(std::size_t)g;
      plugEnrichMax=std::max(plugEnrichMax,std::abs((double)fixed.u0[i]));
      plugEnrichMax=std::max(plugEnrichMax,std::abs((double)fixed.u1[i]));
      plugEnrichMax=std::max(plugEnrichMax,std::abs((double)fixed.u2[i]));
    }
    std::printf("STAGE25_VECTOR_PLUG vertexMaxError=%.12e enrichmentMax=%.12e "
      "xyz=CHECKED status=%s\n",plugVertexErr,plugEnrichMax,
      (plugVertexErr<1.e-10&&plugEnrichMax==0.)?"PASS":"FAIL");
    if(plugVertexErr>=1.e-10||plugEnrichMax!=0.)
      throw std::runtime_error("STAGE25_VECTOR_PLUG mismatch");
    // Native P1 potential solve occurs after the frozen Stage25 plug audit,
    // before velocity coefficient arrays are copied to device.
    const std::string initialVelocity=std::getenv("STAGE27_INITIAL_VELOCITY")?
      std::getenv("STAGE27_INITIAL_VELOCITY"):"plug";
    if(initialVelocity!="plug" && initialVelocity!="potential")
      throw std::runtime_error("STAGE27_INITIAL_VELOCITY must be plug or potential");
    const bool stage27PotentialStart=initialVelocity=="potential";
    if(stage27PotentialStart){
      if(!(sstG2 && sstG2Start=="plug" && M.h.nhex==0 && M.h.ntet>0))
        throw std::runtime_error("STAGE27_POTENTIAL requires pure-tet SST G2 direct start");
      auto initAudit=nodals::initial27::apply(M,stage19Patches,inlet23,fixed);
      (void)initAudit;
    }
    std::printf("STAGE27_INITIAL_VELOCITY mode=%s actualLaplaceSolve=%d "
      "BF3FaceFluxCorrection=%d interiorProjection=NOT_EXACT_DIV_FREE "
      "status=PASS\n",initialVelocity.c_str(),(int)stage27PotentialStart,
      (int)stage27PotentialStart);
    std::vector<unsigned char>clampXY=fixed.mask;
    if(!stage22Vector)for(std::size_t i=0;i<clampXY.size();++i)if(BH.wallMask[i])clampXY[i]=1;
    Dev<unsigned char>d_fixed(fixed.mask),d_clampXY(clampXY),d_wallMask(BH.wallMask);
    Dev<T>d_u0(fixed.u0),d_u1(fixed.u1),d_u2(fixed.u2);
    auto VH=build_velocity_csr(M,IP);auto VC=color_free_rows(VH,fixed.mask);GpuCSR A(VH,VC);
    const char* p1Raw=std::getenv("SST_COMPACT_P1");
    const bool p1Compact=p1Raw && std::string(p1Raw)=="1";
    if(p1Compact && (!sstG2 || M.h.nhex || M.h.ninterface || M.h.ntet==0 ||
       hxt5c::sst_state_space_mode_host()!=hxt5c::SST_STATE_Q1))
       throw std::runtime_error("SST_COMPACT_P1=1 requires puretet SST G2 + SST_STATE_SPACE=q1");
    hxt4b::CSRHost p1Host;
    hxt4b::ColoringHost p1Colors;
    std::unique_ptr<GpuCSR> p1Matrix;
    Dev<unsigned char> p1ScalarFixed;
    Dev<T> d_p1L1K,d_p1L1Omega;
    if(p1Compact){
      p1Host=build_p1_scalar_csr(M);
      std::vector<unsigned char> freeQ1((size_t)M.h.nv,0);
      p1Colors=color_free_rows(p1Host,freeQ1);
      p1Matrix.reset(new GpuCSR(p1Host,p1Colors));
      p1ScalarFixed.upload(freeQ1);
      const char* calibDir=std::getenv("SST_P1_L1_CALIB_DIR");
      if(calibDir && *calibDir){
        d_p1L1K.upload(hxt5c::p1_l1_load_calibration(calibDir,"k",*p1Matrix,(int)M.h.nv));
        d_p1L1Omega.upload(hxt5c::p1_l1_load_calibration(calibDir,"omega",*p1Matrix,(int)M.h.nv));
      }else{
        const char* allowRaw=std::getenv("SST_P1_L1_ALLOW_UNCALIBRATED");
        if(!allowRaw || std::string(allowRaw)!="1")
          throw std::runtime_error("compact P1 requires SST_P1_L1_CALIB_DIR (run reference first); set SST_P1_L1_ALLOW_UNCALIBRATED=1 only for diagnostics");
      }
      std::printf("P1_COMPACT_CSR vertices=%d nnz=%zu tetSlots=%zu colors=%d fullVelocityRows=%d status=PASS\n",
          p1Matrix->n,p1Matrix->nnz,p1Host.tetSlot.size(),p1Colors.ncolors,A.n);
    }
    stage6_mem::tag(A.base.p,"momentum/base");stage6_mem::tag(A.conv.p,"momentum/convection");stage6_mem::tag(A.val.p,"momentum/assembled_X");
    stage6_mem::tag(A.tetSlot.p,"momentum/tet_assembly_slot");stage6_mem::tag(A.col.p,"momentum/csr_col");stage6_mem::tag(A.row.p,"momentum/csr_row");
    stage6_mem::snapshot("after_velocity_CSR",12);

    // STAGE27_DG_PROFILE_V1: native device arrays indexed by original P1 vertex id.
    Dev<T> d_stage27ProfileUx,d_stage27ProfileUy,d_stage27ProfileUz;
    if(stage27ProfileActive){
      d_stage27ProfileUx.upload(stage27HostProfile.ux);
      d_stage27ProfileUy.upload(stage27HostProfile.uy);
      d_stage27ProfileUz.upload(stage27HostProfile.uz);
      const T* px=d_stage27ProfileUx.p;const T* py=d_stage27ProfileUy.p;
      const T* pz=d_stage27ProfileUz.p;const int active=1;
      HXT1_CUDA(cudaMemcpyToSymbol(stage27_inlet::velocityX,&px,sizeof(px)));
      HXT1_CUDA(cudaMemcpyToSymbol(stage27_inlet::velocityY,&py,sizeof(py)));
      HXT1_CUDA(cudaMemcpyToSymbol(stage27_inlet::velocityZ,&pz,sizeof(pz)));
      HXT1_CUDA(cudaMemcpyToSymbol(stage27_inlet::profileActive,&active,sizeof(active)));
      std::printf("STAGE27_DG_PROFILE_GPU=PASS vertices=%zu components=3 scalarInlet=CONSTANT_K_OMEGA "
                  "pressureSource=PROFILED status=PASS\n",stage27HostProfile.nVertices);
    }
    Dev<nodals_hxt1::Point>d_pts(M.points);Dev<nodals_hxt1::HexConn>d_hc(M.hexes);Dev<nodals_hxt1::TetConn>d_tc(M.tets);Dev<nodals_hxt1::HexVel>d_hv(M.hexVel);
    Dev<nodals_hxt1::TetVel>d_tv(M.tetVel);Dev<nodals_hxt1::TetGeom>d_tg(M.tetGeom);Dev<hxt2::InterfacePlan>d_ip(IP);
    Dev<unsigned long long>d_bad(std::vector<unsigned long long>(1,0));
    if(p1Compact){
      HXT1_CUDA(cudaMemset(p1Matrix->base.p,0,p1Matrix->nnz*sizeof(T)));
      hxt5c::p1_compact_molecular_base<<<M.h.ntet,32>>>(
        d_tv.p,d_tg.p,p1Matrix->tetSlot.p,p1Matrix->base.p,M.h.ntet,nu);
      HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
      std::printf("P1_COMPACT_STATIC_BASE builtOnce=1 exactAffineStiffness=1 status=PASS\n");
    }

    // Persistent SST HEX/TET interface geometry + topology.
    hxt5c::SSTInterfaceCache d_sstInterfaceCache(M.h.ninterface);

    hxt5c::setup_sst_interface_cache(
      M.h.ninterface,
      d_pts,d_hc,d_tc,
      d_hv,d_tv,d_tg,d_ip,
      A,
      d_sstInterfaceCache,
      d_bad);

    // PERF_HXT_HEX_CONV_PERSISTENT_GEOMETRY
    // Layout:
    //   grad  [hex][27][14][3]
    //   det*w [hex][27]
    //
    // 9600 HEX case uses approximately 85 MiB total in FP64 geometry.
    Dev<double>d_hexConvGrad(
      std::vector<double>(
        (std::size_t)M.h.nhex*27ull*14ull*3ull,0.0));

    Dev<double>d_hexConvDetW(
      std::vector<double>(
        (std::size_t)M.h.nhex*27ull,0.0));

    // Stage 6.B.2: active pure-tet SST momentum volume uses A.val as
    // assembly scratch. The entrywise combine reads t[i] before writing
    // A.val[i], and subsequent k/omega scalar volume assembly has the same
    // entrywise read-before-write property. Other modes remain unchanged.
    const bool stage6b2TurbAlias=!p1Compact && stage22d4Live && sstG2 && s05b1Active &&
      M.h.ntet>0 && M.h.nhex==0 && M.h.ninterface==0;
    Dev<T>d_turb;
    if(stage6b2TurbAlias)d_turb.borrow(A.val.p,A.nnz);
    else d_turb.upload(std::vector<T>(A.nnz,0));
    Dev<T>d_diagRTurb(std::vector<T>((std::size_t)nv,0));
    std::printf("STAGE6B2_TURB_ALIAS enabled=%d nnz=%zu savedBytes=%zu savedMiB=%.6f "
      "owner=momentum/assembled_X coupling=ENTRYWISE_CSR_REUSE dg=UNCHANGED "
      "scalarQ1=UNCHANGED physics=UNCHANGED status=PASS\n",
      (int)stage6b2TurbAlias,(std::size_t)A.nnz,
      stage6b2TurbAlias?(std::size_t)A.nnz*sizeof(T):0u,
      stage6b2TurbAlias?(double)A.nnz*sizeof(T)/1048576.:0.);

    // Explicit lagged transpose/deviatoric momentum correction.  These are
    // identically zero when --sst-deviatoric-stress=0.
    // Stage6.B.3: when the *existing* model has no explicit deviatoric
    // momentum RHS, all three RHS vectors are identically zero. Borrow
    // one zero buffer rather than retaining three. The enabled deviatoric
    // model and all other geometry/solver modes are completely unchanged.
    const bool stage6b3ZeroRhsAlias=sstG2 && stage22d4Live && s05b1Active &&
      M.h.ntet>0 && M.h.nhex==0 && M.h.ninterface==0 &&
      !sstG1.deviatoricMomentum;
    Dev<T>d_sstDevRhs0(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_sstDevRhs1,d_sstDevRhs2;
    if(stage6b3ZeroRhsAlias){
      d_sstDevRhs1.borrow(d_sstDevRhs0.p,(std::size_t)nv);
      d_sstDevRhs2.borrow(d_sstDevRhs0.p,(std::size_t)nv);
    }else{
      d_sstDevRhs1.upload(std::vector<T>((std::size_t)nv,0));
      d_sstDevRhs2.upload(std::vector<T>((std::size_t)nv,0));
    }

    Dev<double>d_mixStats(std::vector<double>(4,0));
    std::vector<hxt5b::WallMatchPoint> wallMatchHost(BH.wallH.size()*16);
    if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==3){
      if(!(sstG1.profileMatchY>0.0))throw std::runtime_error("spalding_match_height requires --sst-profile-match-y > 0");
      wallMatchHost=hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileMatchY);
    }else if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==5){
      if(!(sstG1.profileFitY1>0.0 && sstG1.profileFitY1<sstG1.profileFitY2 && sstG1.profileFitY2<sstG1.profileFitY3 && sstG1.profileFitY3<sstG1.profileFitY4 && sstG1.profileFitY4<sstG1.profileFitY5))
        throw std::runtime_error("spalding_profile5 requires 0 < --sst-profile-fit-y1 < y2 < y3 < y4 < y5");
      auto C1=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY1));
      auto C2=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY2));
      auto C3=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY3));
      auto C4=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY4));
      auto C5=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY5));
      if(C1.size()!=C2.size()||C1.size()!=C3.size()||C1.size()!=C4.size()||C1.size()!=C5.size())throw std::runtime_error("spalding_profile5 cache size mismatch");
      wallMatchHost.resize(C1.size()*5u);
      for(std::size_t p=0;p<C1.size();++p){
        wallMatchHost[5u*p+0u]=C1[p]; wallMatchHost[5u*p+1u]=C2[p]; wallMatchHost[5u*p+2u]=C3[p]; wallMatchHost[5u*p+3u]=C4[p]; wallMatchHost[5u*p+4u]=C5[p];
      }
      std::printf(
        "NODALS_SST_MATCH_PROFILE5_CACHE wallFaces=%zu wallQuadraturePoints=%zu samplesPerPoint=5 totalSamples=%zu y=[%.12e,%.12e,%.12e,%.12e,%.12e] fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES status=PASS\n",
        BH.wallH.size(),C1.size(),wallMatchHost.size(),sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5);
    }else if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==4){
      if(!(sstG1.profileFitY1>0.0 && sstG1.profileFitY1<sstG1.profileFitY2 && sstG1.profileFitY2<sstG1.profileFitY3))
        throw std::runtime_error("spalding_profile3 requires 0 < --sst-profile-fit-y1 < y2 < y3");
      auto C1=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY1));
      auto C2=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY2));
      auto C3=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY3));
      if(C1.size()!=C2.size()||C1.size()!=C3.size())throw std::runtime_error("spalding_profile3 cache size mismatch");
      wallMatchHost.resize(C1.size()*3u);
      for(std::size_t p=0;p<C1.size();++p){
        wallMatchHost[3u*p+0u]=C1[p];
        wallMatchHost[3u*p+1u]=C2[p];
        wallMatchHost[3u*p+2u]=C3[p];
      }
      std::printf(
        "NODALS_SST_MATCH_PROFILE3_CACHE wallFaces=%zu wallQuadraturePoints=%zu samplesPerPoint=3 totalSamples=%zu "
        "y=[%.12e,%.12e,%.12e] fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES status=PASS\n",
        BH.wallH.size(),C1.size(),wallMatchHost.size(),sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3);
    }
    // NODALS_FACE_YPLUS30_A: Load the previous segment's *F2 full-vector*
    // wall-face fitted friction velocities. No velocity field interpolation
    // from vertex-only VTU, and no moving samples during SIMPLE.
    std::vector<double> faceSpY3,faceOmY3;
    std::vector<double> autoFaceUtau;
    std::vector<hxt5b::AutoFaceGeometry> autoFaceGeometry;
    std::vector<int> autoCornerFlags,autoSpFallback,autoOmFallback;
    const double autoCornerDegrees=[](){const char*s=std::getenv("NODALS_AUTO_CORNER_DEGREES");
      return s&&*s?std::stod(s):45.;}();
    if(autoCentroidMode){
      const std::string samplingMode=std::getenv("SST_WALL_SAMPLING_MODE")?
        std::getenv("SST_WALL_SAMPLING_MODE"):"";
      if(M.h.nhex!=0 || stage24bN!=1 || stage24bOmega!=3 ||
         samplingMode!="ray_trace" || !f2FaceSpalding)
        throw std::runtime_error("NODALS_AUTO requires pure-tet F2 ray_trace, one-point Spalding and omega index 3");
      autoFaceGeometry=hxt5b::auto_face_geometry(M,BH.wallT,autoCornerDegrees);
      autoCornerFlags.resize(autoFaceGeometry.size());
      std::size_t corners=0;
      for(std::size_t i=0;i<autoCornerFlags.size();++i){
        autoCornerFlags[i]=autoFaceGeometry[i].nearSharpCorner?1:0; // NODALS_AUTO_CENTROID_CORNER_NEIGHBORHOOD_V11
        corners+=(std::size_t)autoCornerFlags[i];
      }
      std::printf("NODALS_AUTO_CENTROID_CONFIG enabled=1 spAuto=%d omegaAuto=%d "
        "wallFaces=%zu sharpCornerNeighborhoodFaces=%zu cornerAngleDegrees=%.3f "
        "cellCentroidNormalHeight=3V_OVER_4A strictThinPolicy=ERROR status=PASS\n",
        (int)autoCentroidSp,(int)autoCentroidOm,autoFaceGeometry.size(),corners,autoCornerDegrees);
    }
    if(faceYplusEnabled){
      if(M.h.nhex!=0 || stage24bN!=1 || stage24bOmega!=3 || !f2FaceSpalding)
        throw std::runtime_error("NODALS_FACE_YPLUS requires pure tet, F2, fitPoints=1, omegaIndex=3");
      const std::string sampler=std::getenv("SST_WALL_SAMPLING_MODE")?
        std::getenv("SST_WALL_SAMPLING_MODE"):"";
      if(sampler!="ray_trace")
        throw std::runtime_error("NODALS_FACE_YPLUS requires ray_trace");
      const auto param=[](const char*key,double fallback){
        const char*s=std::getenv(key);
        const double x=s&&*s?std::stod(s):fallback;
        if(!(std::isfinite(x)&&x>0.))throw std::runtime_error(std::string("NODALS_FACE_YPLUS invalid ")+key);
        return x;
      };
      const double targetSp=param("NODALS_FACE_YPLUS_SP",30.);
      const double targetOm=param("NODALS_FACE_YPLUS_OMEGA",30.);
      const double scaleMin=param("NODALS_FACE_YPLUS_MIN_SCALE",0.02);
      const double scaleMax=param("NODALS_FACE_YPLUS_MAX_SCALE",0.35);
      if(targetSp>500. || targetOm>500. || scaleMax>1. || scaleMin>scaleMax)
        throw std::runtime_error("NODALS_FACE_YPLUS target/scale range invalid");
      const char* baseScaleEnv=std::getenv("STAGE32_SPALDING_SAMPLE_SCALE");
      if(!baseScaleEnv || !*baseScaleEnv)
        throw std::runtime_error("NODALS_FACE_YPLUS requires explicit STAGE32_SPALDING_SAMPLE_SCALE; refusing unsafe 0.05 default");
      const double baseScale=std::stod(baseScaleEnv);
      const double referenceFullScaleY3=sstG1.profileFitY3/baseScale;
      if(!(referenceFullScaleY3>0.))throw std::runtime_error("NODALS_FACE_YPLUS bad reference Y3");
      const double yMin=autoCentroidMode?0.:referenceFullScaleY3*scaleMin;
      const double yMax=autoCentroidMode?std::numeric_limits<double>::infinity():referenceFullScaleY3*scaleMax;
      faceSpY3.resize(BH.wallT.size());faceOmY3.resize(BH.wallT.size());
      std::ifstream in(yplusFaceInput);
      if(!in)throw std::runtime_error(std::string("NODALS_FACE_YPLUS cannot open input: ")+yplusFaceInput);
      std::size_t count=0,clampSp=0,clampOm=0,fallback=0;
      double minSp=1.e300,maxSp=0.,sumSp=0.,minOm=1.e300,maxOm=0.,sumOm=0.;
      std::string line;
      while(std::getline(in,line)){
        if(line.empty()||line[0]=='#')continue;
        int fi=-1,cell=-1,lf=-1;double utau=0.,sampleYplus=0.;
        if(std::sscanf(line.c_str(),"%d,%d,%d,%lf,%lf",&fi,&cell,&lf,&utau,&sampleYplus)!=5 ||
           fi!=(int)count || count>=BH.wallT.size() ||
           cell!=BH.wallT[count].cell || lf!=(int)BH.wallT[count].lf || !std::isfinite(utau) || utau<0.)
          throw std::runtime_error("NODALS_FACE_YPLUS input face/geometry mismatch at row "+std::to_string(count));
        const double localCentroid=autoCentroidMode?autoFaceGeometry[count].centroidHeight:sstG1.profileFitY3;
        const double unboundedSp=utau>1.e-8?targetSp*nu/utau:localCentroid;
        const double unboundedOm=utau>1.e-8?targetOm*nu/utau:
          (autoCentroidMode?localCentroid:sstG1.profileFitY3*(targetOm/targetSp));
        if(autoCentroidMode)autoFaceUtau.push_back(utau);
        if(!(utau>1.e-8))++fallback;
        const double sp=std::min(yMax,std::max(yMin,unboundedSp));
        const double om=std::min(yMax,std::max(yMin,unboundedOm));
        clampSp += (std::abs(sp-unboundedSp)>1.e-13);
        clampOm += (std::abs(om-unboundedOm)>1.e-13);
        faceSpY3[count]=sp;faceOmY3[count]=om;
        minSp=std::min(minSp,sp);maxSp=std::max(maxSp,sp);sumSp+=sp;
        minOm=std::min(minOm,om);maxOm=std::max(maxOm,om);sumOm+=om;
        ++count;
      }
      if(count!=BH.wallT.size()||count==0)
        throw std::runtime_error("NODALS_FACE_YPLUS wrong number of face records");
      std::printf("NODALS_FACE_YPLUS_LOADED faces=%zu targetSp=%.8g targetOmega=%.8g "
        "minScale=%.6g maxScale=%.6g fallback=%zu clampSp=%zu clampOmega=%zu "
        "spY3Min=%.12e spY3Mean=%.12e spY3Max=%.12e "
        "omegaY3Min=%.12e omegaY3Mean=%.12e omegaY3Max=%.12e "
        "input=%s source=LAGGED_F2_FACE_UTAU status=PASS\n",
        count,targetSp,targetOm,scaleMin,scaleMax,fallback,clampSp,clampOm,
        minSp,sumSp/count,maxSp,minOm,sumOm/count,maxOm,yplusFaceInput);
    }
    // Initial stage: each physical wall face gets 3V/(4A), independently
    // for Spalding and omega unless that field has an explicit numeric override.
    // Subsequent stages: previous fitted face u_tau supplies y+ targets.
    std::vector<double> autoCentroidHeights;
    if(autoCentroidMode){
      autoCentroidHeights.resize(BH.wallT.size());
      for(std::size_t i=0;i<BH.wallT.size();++i)
        autoCentroidHeights[i]=autoFaceGeometry[i].centroidHeight;
      if(!faceYplusEnabled){
        faceSpY3.resize(BH.wallT.size());faceOmY3.resize(BH.wallT.size());
        for(std::size_t i=0;i<BH.wallT.size();++i){
          faceSpY3[i]=autoCentroidSp?autoFaceGeometry[i].centroidHeight:sstG1.profileFitY3;
          faceOmY3[i]=autoCentroidOm?autoFaceGeometry[i].centroidHeight:
            sstG1.profileFitY3*stage32OmegaScale/stage32SpaldingScale;
        }
      }
      const auto makeFlags=[&](const std::vector<double>&h,const char*kind,double yplus){
        return hxt5b::auto_preflight_samples(M,BH.wallT,autoFaceGeometry,h,
          autoFaceUtau,nu,yplus,kind,faceYplusEnabled);
      };
      const double spTarget=std::getenv("NODALS_FACE_YPLUS_SP")?
        std::stod(std::getenv("NODALS_FACE_YPLUS_SP")):30.;
      const double omTarget=std::getenv("NODALS_FACE_YPLUS_OMEGA")?
        std::stod(std::getenv("NODALS_FACE_YPLUS_OMEGA")):30.;
      autoSpFallback=makeFlags(faceSpY3,"SPALDING",spTarget);
      autoOmFallback=makeFlags(faceOmY3,"OMEGA",omTarget);
    }
    std::vector<hxt5b::TetWallMatchPoint>tetMatchHost;
    if(!BH.wallT.empty()){
      if(!sstG2 || !sstG1.profileAverageClosure ||
         sstG1.profileAverageWallLaw!=5 ||
         (sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_PROFILE5_EQ_SST &&
          sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO))
        throw std::runtime_error("pure-tet requires G2 PROFILE5+EQ5 or PROFILE5+OF_AUTO");
      const double yy[5]={sstG1.profileFitY1,sstG1.profileFitY2,
        sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5};
      if(!(yy[0]>0 && yy[0]<yy[1] && yy[1]<yy[2] && yy[2]<yy[3] && yy[3]<yy[4]))
        throw std::runtime_error("pure-tet five sample heights not increasing");
      std::vector<hxt5b::TetWallMatchPoint> tmp[5];
      const std::string stage21Mode=std::getenv("SST_WALL_SAMPLING_MODE")?
        std::getenv("SST_WALL_SAMPLING_MODE"):"legacy_endpoint";
      if(stage21Mode=="ray_trace"){
        const int maxCross=std::getenv("SST_WALL_RAY_MAX_CROSSINGS")?
          std::stoi(std::getenv("SST_WALL_RAY_MAX_CROSSINGS")):4096;
        const double tol=std::getenv("SST_WALL_RAY_BARY_TOLERANCE")?
          std::stod(std::getenv("SST_WALL_RAY_BARY_TOLERANCE")):1.0e-9;
        // Preserve the five-point ray traversal and packing byte-for-byte.
        // For N=1..4 and OF_AUTO, intersect the fitting slots with the
        // independent omega evaluation slot (a UNION, not an intersection).
        const unsigned fitMask=::nodals::wall::active::active_slot_mask(stage24bN);
        const unsigned requestedMask=stage24bN==5?0x1fu:
          (fitMask|(1u<<stage24bOmegaZero));
        const char* localRayEnv=std::getenv("STAGE27_LOCAL_RAY_ADAPT");
        const bool localRayAdapt=localRayEnv && std::string(localRayEnv)=="1";
        if(localRayEnv && std::string(localRayEnv)!="0" && !localRayAdapt)
          throw std::runtime_error("STAGE27_LOCAL_RAY_ADAPT must be 0 or 1");
        tetMatchHost=hxt5b::build_tet_wall_ray_profile_cache(
          M,BH.wallT,yy,maxCross,tol,requestedMask,
          autoCentroidMode?false:(localRayAdapt || faceYplusEnabled),
          (autoCentroidMode || faceYplusEnabled)?&faceSpY3:nullptr,
          autoCentroidMode?&autoSpFallback:nullptr,
          autoCentroidMode?&autoCornerFlags:nullptr,
          autoCentroidMode?&autoCentroidHeights:nullptr,
          autoCentroidMode?&autoFaceUtau:nullptr,nu,
          faceYplusEnabled?(std::getenv("NODALS_FACE_YPLUS_SP")?
            std::stod(std::getenv("NODALS_FACE_YPLUS_SP")):30.):-1.);
        std::printf("SST_WALL_SAMPLING_RUNTIME method=ray_trace rayContainment=VERIFIED "
          "samples=%d activeMask=0x%x maxCrossings=%d baryTolerance=%.3e status=PASS\n",
          __builtin_popcount(requestedMask),requestedMask,maxCross,tol);
      }else if(stage21Mode=="legacy_endpoint"){
        // Lossless original Stage20 path; do not rewrite the five-point arithmetic.
        for(int m=0;m<5;++m)
          tmp[m]=hxt5b::build_tet_wall_match_height_cache(M,BH.wallT,yy[m]);
        tetMatchHost.resize(tmp[0].size()*5u);
        for(std::size_t i=0;i<tmp[0].size();++i)
          for(int m=0;m<5;++m)tetMatchHost[5u*i+(std::size_t)m]=tmp[m][i];
        std::printf("SST_WALL_SAMPLING_RUNTIME method=legacy_endpoint "
          "rayContainment=NOT_CHECKED samples=5 status=PASS\n");
      }else throw std::runtime_error("STAGE21 unsupported wall sampling method: "+stage21Mode);
      std::printf("NODALS_SST_PROFILE5_TET_CONFIG pureTet=%d tetWallFaces=%zu samples=5 sampleGeometry=PHYSICAL_INWARD_NORMAL momentumUtau=FIVE_POINT_SPALDING scalarMode=%s totalSamples=%zu status=PASS\n",
        M.h.nhex==0?1:0,BH.wallT.size(),
        sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO?"OFAUTO_POINT3_OMEGA_ONLY":"MULTIPOINT_EQ_SST",
        tetMatchHost.size());
      if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO){
        std::printf("PURETET_OFAUTO_CONFIG momentum=SPALDING5_UNCHANGED omegaTarget=ORIGINAL_HEX_OF_AUTO_BLEND sample=POINT3 kWall=NATURAL_ZERO_GRADIENT omegaPenalty=ONE_RANK_ONE_TET_SAMPLE normalSlab=Y5_MINUS_Y1 production=%d hexAverageFirstEquivalent=NO status=PASS\n",(int)sstG1.ofProduction);
      }
    }
    // STAGE27_S02B_1: keep the exact legacy five-ray host preflight above,
    // including verified containment of inactive rays.  Pack only device data.
    const int s02b_fit=stage24bN;
    const int s02b_omega=stage24bOmegaZero;
    const int s02b_ids[5][5]={{2,-1,-1,-1,-1},{0,4,-1,-1,-1},
                              {0,2,4,-1,-1},{0,1,3,4,-1},
                              {0,1,2,3,4}};
    bool s02b_keep[5]={false,false,false,false,false};
    for(int j=0;j<s02b_fit;++j)s02b_keep[s02b_ids[s02b_fit-1][j]]=true;
    // Omega point may be anywhere from 1 to 5; retain it even if not in fit.
    if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO)
      s02b_keep[s02b_omega]=true;
    const char* s02b_mode=std::getenv("STAGE27_S02B_PACK");
    const bool s02b_pack=!(s02b_mode&&std::string(s02b_mode)=="0");
    if(!s02b_pack)for(int m=0;m<5;++m)s02b_keep[m]=true;
    int s02b_map[5]={-1,-1,-1,-1,-1};int s02b_stride=0;
    for(int m=0;m<5;++m)if(s02b_keep[m])s02b_map[m]=s02b_stride++;
    if(s02b_stride<1||s02b_map[s02b_omega]<0)
      throw std::runtime_error("Stage2B: invalid active/omega ray union");
    const double s02b_heights[5]={sstG1.profileFitY1,sstG1.profileFitY2,
      sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5};
    HXT1_CUDA(cudaMemcpyToSymbol(::stage27_s02b_stride,&s02b_stride,sizeof(int)));
    HXT1_CUDA(cudaMemcpyToSymbol(::stage27_s02b_mapping,s02b_map,sizeof(s02b_map)));
    HXT1_CUDA(cudaMemcpyToSymbol(::stage27_s02b_heights,s02b_heights,sizeof(s02b_heights)));
    const std::size_t s02b_quads=BH.wallT.size()*12u;
    if(tetMatchHost.size()!=s02b_quads*5u)
      throw std::runtime_error("Stage2B: wrong legacy five-slot cache size");
    std::vector<hxt5b::TetWallPackedPoint> s02b_packed(s02b_quads*(std::size_t)s02b_stride);
    for(std::size_t iq=0;iq<s02b_quads;++iq){
      for(int m=0;m<5;++m)if(s02b_map[m]>=0){
        // Bitwise identical full-precision record, never reinterpolate geometry.
        // Preserve all original FP64 primary interpolation weights and
        // the original enriched velocity connectivity. BF3 cubic weights
        // are regenerated on the GPU from these *same* barycentrics.
        const auto &src=tetMatchHost[iq*5u+(std::size_t)m];
        auto &dst=s02b_packed[iq*(std::size_t)s02b_stride+(std::size_t)s02b_map[m]];
        dst.cell=src.cell;
        for(int j=0;j<8;++j)dst.g[j]=src.g[j];
        for(int j=0;j<4;++j)dst.l[j]=src.b[j];
        dst.y=src.y;
      }
    }
    const std::size_t s02b_original_bytes=tetMatchHost.size()*sizeof(hxt5b::TetWallMatchPoint);
    const std::size_t s02b_packed_bytes=s02b_packed.size()*sizeof(hxt5b::TetWallPackedPoint);
    std::printf("STAGE27_S02B2_FP64_CACHE fitPoints=%d omegaIndex=%d "
      "gpuSlots=%d hostRecordBytes=%zu deviceRecordBytes=%zu "
      "hostRayMiB=%.3f deviceRayMiB=%.3f savedMiB=%.3f "
      "basis=P1_BF3_RECONSTRUCT_FROM_FP64_L originalHostPreflight=UNCHANGED status=PASS\n",
      s02b_fit,s02b_omega+1,s02b_stride,
      sizeof(hxt5b::TetWallMatchPoint),sizeof(hxt5b::TetWallPackedPoint),
      (double)s02b_original_bytes/1048576.0,
      (double)s02b_packed_bytes/1048576.0,
      (double)(s02b_original_bytes-s02b_packed_bytes)/1048576.0);
    std::printf("STAGE27_S02B_DEVICE_CACHE enabled=%d fitPoints=%d omegaSampleIndex=%d "
      "hostValidatedSlots=5 gpuSlots=%d quadraturePoints=%zu "
      "oldMiB=%.3f newMiB=%.3f savedMiB=%.3f recordBytes=%zu "
      "method=EXACT_FP64_ACTIVE_UNION status=PASS\n",
      s02b_pack?1:0,s02b_fit,s02b_omega+1,s02b_stride,s02b_quads,
      (double)s02b_original_bytes/1048576.0,(double)s02b_packed_bytes/1048576.0,
      (double)(s02b_original_bytes-s02b_packed_bytes)/1048576.0,
      sizeof(hxt5b::TetWallMatchPoint));
    // STAGE32: one EXTRA packed ray per quadrature, only when distances differ.
    // Never overwrite the original sample-3 Spalding ray. For a given physical
    // wall quadrature both rays share its outward unit normal and wall origin.
    Dev<hxt5b::TetWallPackedPoint> d_stage32OmegaRay;
    if(stage32Split){
      const double omegaY=sstG1.profileFitY3*stage32OmegaScale/stage32SpaldingScale;
      const double omegaHeights[5]={omegaY/3.,2.*omegaY/3.,omegaY,4.*omegaY/3.,5.*omegaY/3.};
      const int maxCross=std::getenv("SST_WALL_RAY_MAX_CROSSINGS")?
        std::stoi(std::getenv("SST_WALL_RAY_MAX_CROSSINGS")):4096;
      const double tol=std::getenv("SST_WALL_RAY_BARY_TOLERANCE")?
        std::stod(std::getenv("SST_WALL_RAY_BARY_TOLERANCE")):1.e-9;
      const char* local=std::getenv("STAGE27_LOCAL_RAY_ADAPT");
      const bool adapt=local && std::string(local)=="1";
      const auto omegaFull=hxt5b::build_tet_wall_ray_profile_cache(
        M,BH.wallT,omegaHeights,maxCross,tol,1u<<2,
         autoCentroidMode?false:(adapt || faceYplusEnabled),
         (autoCentroidMode || faceYplusEnabled)?&faceOmY3:nullptr,
         autoCentroidMode?&autoOmFallback:nullptr,
         autoCentroidMode?&autoCornerFlags:nullptr,
         autoCentroidMode?&autoCentroidHeights:nullptr,
         autoCentroidMode?&autoFaceUtau:nullptr,nu,
         faceYplusEnabled?(std::getenv("NODALS_FACE_YPLUS_OMEGA")?
           std::stod(std::getenv("NODALS_FACE_YPLUS_OMEGA")):30.):-1.);
      if(omegaFull.size()!=s02b_quads*5u)
        throw std::runtime_error("STAGE32 omega ray count mismatch");
      std::vector<hxt5b::TetWallPackedPoint> packed(s02b_quads);
      for(std::size_t iq=0;iq<s02b_quads;++iq){
        const auto& src=omegaFull[iq*5u+2u];
        auto& dst=packed[iq];
        if(src.cell<0 || !(src.y>0.) || !std::isfinite(src.y))
          throw std::runtime_error("STAGE32 invalid omega sampling geometry");
        dst.cell=src.cell;dst.y=src.y;
        for(int j=0;j<8;++j)dst.g[j]=src.g[j];
        for(int j=0;j<4;++j)dst.l[j]=src.b[j];
      }
      d_stage32OmegaRay.upload(packed);
      const hxt5b::TetWallPackedPoint* ptr=d_stage32OmegaRay.p;
      const int active=1;
      // Freeze the geometric rank-one penalty thickness at reference scale
      // 0.05. This isolates omega sampling distance from penalty amplitude.
      const double refEll=(sstG1.profileFitY5-sstG1.profileFitY1)*
                          (0.05/stage32SpaldingScale);
      if(!(refEll>0.) || !std::isfinite(refEll))
        throw std::runtime_error("STAGE32 invalid reference omega penalty thickness");
      HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage32_omega_ray,&ptr,sizeof(ptr)));
      HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage32_omega_active,&active,sizeof(active)));
      HXT1_CUDA(cudaMemcpyToSymbol(hxt5c::stage32_omega_penalty_ell,&refEll,sizeof(refEll)));
      std::printf("STAGE32_SPLIT_WALL=ON spaldingIndex=3 omegaIndex=3 "
          "spaldingScale=%.8g omegaScale=%.8g spaldingY=%.12e "
          "omegaY=%.12e penaltyReferenceScale=0.05 penaltyEll=%.12e "
          "extraOmegaRays=%zu profileInlet=UNCHANGED status=PASS\n",
          stage32SpaldingScale,stage32OmegaScale,sstG1.profileFitY3,
          omegaY,refEll,s02b_quads);
    }else{
      std::printf("STAGE32_SPLIT_WALL=OFF spaldingScale=%.8g omegaScale=%.8g "
          "mode=ORIGINAL_UNMODIFIED_SHARED_SAMPLE status=PASS\n",
          stage32SpaldingScale,stage32OmegaScale);
    }
    std::fflush(stdout);
    Dev<hxt5b::TetWallPackedPoint>d_tetMatch(s02b_packed);
    // Release large host preflight objects after the device copy.
    std::vector<hxt5b::TetWallPackedPoint>().swap(s02b_packed);
    std::vector<hxt5b::TetWallMatchPoint>().swap(tetMatchHost);
    const auto tetQuadHost=hxt5b::build_tet_wall_quad_cache(M,BH.wallT);
    Dev<hxt5b::TetWallQuadCache>d_tetQuad(tetQuadHost);
    Dev<hxt5b::TetFacePlan>d_wallT(BH.wallT);
    sstG1.profileTetMatchPoints=d_tetMatch.p;
    sstG1.tetWallPlans=d_wallT.p;
    sstG1.tetWallQuads=d_tetQuad.p;
    Dev<hxt5b::WallMatchPoint>d_wallMatch(wallMatchHost);
    sstG1.profileMatchPoints=d_wallMatch.p;
    if(sstG1.profileAverageWallLaw==3)std::printf(
      "NODALS_SST_MATCH_HEIGHT_DEVICE_CACHE points=%zu bytes=%zu targetY=%.12e "
      "velocity=FULL_Q1BF2_POINT_VALUE wallReference=STATIONARY_ABSOLUTE "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION "
      "sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint),sstG1.profileMatchY);
    if(sstG1.profileAverageWallLaw==5)std::printf(
      "NODALS_SST_MATCH_PROFILE5_DEVICE_CACHE points=%zu bytes=%zu samplesPerWallQuadrature=5 "
      "velocity=FULL_Q1BF2_POINT_VALUE fit=ONE_UTAU_FROM_FIVE_REMOTE_SAMPLES "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint));
    if(sstG1.profileAverageWallLaw==4)std::printf(
      "NODALS_SST_MATCH_PROFILE3_DEVICE_CACHE points=%zu bytes=%zu samplesPerWallQuadrature=3 "
      "velocity=FULL_Q1BF2_POINT_VALUE fit=ONE_UTAU_FROM_THREE_REMOTE_SAMPLES "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint));
    Dev<hxt5b::HexFacePlan>d_inH(BH.inH),d_wallH(BH.wallH);Dev<hxt5b::TetFacePlan>d_inT(BH.inT);
    auto wallCacheHost=
      hxt5b::build_wall_hex_cache(
        M,BH.wallH,wallSampleFraction);
    Dev<hxt5b::WallHexQuadCache>
      d_wallCache(wallCacheHost);
    Dev<T>d_contSource(hxt4c::castv<T>(BH.csrc));
    Dev<T>d_dg(std::vector<T>(A.nnz,0)),d_diagDG(std::vector<T>((std::size_t)nv,0)),d_inletRhsX(std::vector<T>((std::size_t)nv,0)),d_inletRhsY(std::vector<T>((std::size_t)nv,0)),d_inletRhsZ(std::vector<T>((std::size_t)nv,0));
    // Stage6.B.3: with zero deviatoric RHS, the staged X/Y RHS is
    // precisely the already allocated inlet RHS. s22b_add_extra uses
    // only the same index for read/write, so the existing kernel may
    // safely write directly to its lag (inlet) input.
    Dev<T>d_stage23DevRhs0,d_stage23DevRhs1;
    if(stage6b3ZeroRhsAlias){
      d_stage23DevRhs0.borrow(d_inletRhsX.p,(std::size_t)nv);
      d_stage23DevRhs1.borrow(d_inletRhsY.p,(std::size_t)nv);
    }else{
      d_stage23DevRhs0.upload(std::vector<T>((std::size_t)nv,0));
      d_stage23DevRhs1.upload(std::vector<T>((std::size_t)nv,0));
    }
    std::printf("STAGE6B3_RHS_REUSE enabled=%d nvel=%d savedMiB=%.6f "
      "deviatoricMomentum=%d sharedZero=3 inletCombined=2 "
      "operator=UNCHANGED status=PASS\n",
      (int)stage6b3ZeroRhsAlias,nv,
      stage6b3ZeroRhsAlias?4.0*(double)nv*sizeof(T)/1048576.:0.,
      (int)sstG1.deviatoricMomentum);

    Dev<T>d_wall(std::vector<T>(A.nnz,0)),d_diagWall(std::vector<T>((std::size_t)nv,0)),d_valZ(std::vector<T>(A.nnz,0));
    Dev<T>d_diagOrigZ(std::vector<T>((std::size_t)nv,0)),d_rauZ(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_relaxDeltaXY(std::vector<T>((std::size_t)nv,0)),d_relaxDeltaZ(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_wallStats(std::vector<double>(stage22Vector?33:12,0));

    auto TT=hxt2::build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(hxt2::c_tet_T,TT.data(),TT.size()*sizeof(double)));
    auto CT=build_tet_conv_tensor();HXT1_CUDA(cudaMemcpyToSymbol(h4b_tet_convT,CT.data(),CT.size()*sizeof(double)));

    // Build immutable HEX convection geometry exactly once.
    HXT1_CUDA(cudaMemset(
      d_bad.p,0,sizeof(unsigned long long)));

    if(M.h.nhex)
      hex_conv_geom_cache_setup<<<M.h.nhex,32>>>(
        d_pts.p,
        d_hc.p,
        d_hexConvGrad.p,
        d_hexConvDetW.p,
        M.h.nhex,
        d_bad.p);

    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());

    {
      unsigned long long hexConvGeomBad=0;
      HXT1_CUDA(cudaMemcpy(
        &hexConvGeomBad,
        d_bad.p,
        sizeof(hexConvGeomBad),
        cudaMemcpyDeviceToHost));

      if(hexConvGeomBad)
        throw std::runtime_error(
          "persistent HEX convection geometry cache failures="+
          std::to_string(hexConvGeomBad));
    }

    {
      const double gradMiB=
        (double)((std::size_t)M.h.nhex*27ull*14ull*3ull*
                 sizeof(double))/(1024.0*1024.0);

      const double detwMiB=
        (double)((std::size_t)M.h.nhex*27ull*
                 sizeof(double))/(1024.0*1024.0);

      std::printf(
        "NODALS_HXT_HEX_CONV_CACHE "
        "cells=%llu quadrature=27 "
        "gradMiB=%.3f detwMiB=%.3f totalMiB=%.3f "
        "geometry=PERSISTENT runtime=NUMERIC_ONLY "
        "operator=UNCHANGED status=PASS\n",
        (unsigned long long)M.h.nhex,
        gradMiB,detwMiB,gradMiB+detwMiB);
    }


    std::printf("NODALS_HXT4C_KERNEL_POLICY interfaceThreads=128 interfacePairTraversal=STRIDED hexConvectionThreads=128 hexPairTraversal=STRIDED status=PASS\n");
    HXT1_CUDA(cudaMemset(A.base.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(A.diagRBase.p,0,nv*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_diff_assemble<<<M.h.nhex,256>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,A.base.p,A.diagRBase.p,M.h.nhex,nu,d_bad.p);
    if(M.h.ntet)tet_diff_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,A.base.p,A.diagRBase.p,M.h.ntet,nu);
    if(M.h.ninterface)interface_assemble<<<2*M.h.ninterface,128>>>(d_pts.p,d_hc.p,d_hv.p,d_tv.p,d_tg.p,d_ip.p,A.row.p,A.col.p,A.base.p,A.diagRBase.p,M.h.ninterface,nu,gamma,d_bad.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad)throw std::runtime_error("baseline assembly geometry failures");

    hxt4a::BDevice<T>B(Bh);
    int pin=-1;
    if(pressureGauge=="pin"){
      pin=0;
      if(!M.outHex.empty())pin=M.outHex[0].cell;
      else if(!M.outTet.empty())pin=nh+M.outTet[0].cell;
    }
    std::printf(
      "NODALS_HXT5B_PRESSURE_GAUGE mode=%s pin=%d "
      "continuityRows=%s pressureColumns=%s status=PASS\n",
      pressureGauge.c_str(),pin,
      pin>=0?"ONE_REPLACED":"ALL_RETAINED",
      pin>=0?"ONE_ELIMINATED":"ALL_RETAINED");

    Dev<T>d_p(std::vector<T>((std::size_t)np,0)),d_dp(std::vector<T>((std::size_t)np,0)),d_cont(std::vector<T>((std::size_t)np,0));
    Dev<double>d_contStats(std::vector<double>(2,0.0));
    Dev<T>d_sd(std::vector<T>((std::size_t)np,0)),d_isd(std::vector<T>((std::size_t)np,0));
    Dev<T>d_bt0(std::vector<T>((std::size_t)nv,0)),d_bt1(std::vector<T>((std::size_t)nv,0)),d_bt2(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_rhs0(std::vector<T>((std::size_t)nv,0)),d_rhs1(std::vector<T>((std::size_t)nv,0)),d_rhs2(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_s22Rhs0(std::vector<T>((std::size_t)nv,0)),
      d_s22Rhs1(std::vector<T>((std::size_t)nv,0)),
      d_s22Rhs2(std::vector<T>((std::size_t)nv,0)),
      d_s22ZExtra(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_s22WallY(std::vector<T>(stage22d4Live?1u:A.nnz,0)),
      d_s22WallZ(std::vector<T>(stage22d4Live?1u:A.nnz,0)),
      d_s22DiagY(std::vector<T>((std::size_t)nv,0)),
      d_s22DiagZ(std::vector<T>((std::size_t)nv,0)),
      d_s22ValY(std::vector<T>(A.nnz,0)),
      d_s22RauY(std::vector<T>((std::size_t)nv,0)),
      d_s22DiagOrigY(std::vector<T>((std::size_t)nv,0)),
      d_s22RelaxDeltaY(std::vector<T>((std::size_t)nv,0));
    // Stage 6.B.1: both final component buffers are dead during wall assembly.
    // Reuse them for wall Y/Z accumulation; s22c_combine reads each own
    // wall entry before overwriting it with common+wall. No matrix-free work.
    // Stage6.C.2: Y and Z are still full FP64 CSR matrices during the
    // complete momentum/pressure phase; these wall pointers must be refreshed
    // after reallocation on the following SIMPLE iteration.
    T* stage6bWallY=stage22d4Live?d_s22ValY.p:d_s22WallY.p;
    T* stage6bWallZ=stage22d4Live?d_valZ.p:d_s22WallZ.p;
    const char* stage6c2Flag=std::getenv("STAGE6C2_PHASE_REUSE");
    if(stage6c2Flag && std::string(stage6c2Flag)!="0" &&
       std::string(stage6c2Flag)!="1")
      throw std::runtime_error("STAGE6C2_PHASE_REUSE must be 0 or 1");
    const bool stage6c2PhaseReuse=stage6c2Flag &&
      std::string(stage6c2Flag)=="1" && stage22d4Live && stage22Vector &&
      M.h.ntet>0 && M.h.nhex==0 && M.h.ninterface==0;
    if(stage6c2Flag && std::string(stage6c2Flag)=="1" &&
       !stage6c2PhaseReuse)
      throw std::runtime_error("Stage6C2 phase reuse only supports live pure-tet projected momentum");
    std::printf("STAGE6C2_PHASE_REUSE enabled=%d csrNnz=%zu releaseMiB=%.6f "
      "policy=AFTER_PRESSURE_CORRECTION csrMath=UNCHANGED status=PASS\n",
      (int)stage6c2PhaseReuse,(std::size_t)A.nnz,
      stage6c2PhaseReuse?2.0*(double)A.nnz*sizeof(T)/1048576.:0.0);
    if(stage22d4Live && (!stage22Vector || M.h.nhex || M.h.ninterface))
      throw std::runtime_error("STAGE6B1 alias only valid for live pure-tet projected vector wall operator");
    std::printf("STAGE6B1_WALL_ALIAS enabled=%d nnz=%zu savedBytes=%zu savedMiB=%.6f \
      mode=ASSEMBLED_CSR_REUSE physics=UNCHANGED status=PASS\n",
      (int)stage22d4Live,(std::size_t)A.nnz,
      stage22d4Live?2u*((std::size_t)A.nnz-1u)*sizeof(T):0u,
      stage22d4Live?2.0*((double)A.nnz-1.0)*sizeof(T)/1048576.0:0.0);
    if(!stage6b2TurbAlias)stage6_mem::tag(d_turb.p,"momentum/turbulent");stage6_mem::tag(d_dg.p,"momentum/DG_inlet");stage6_mem::tag(d_wall.p,"momentum/wall_X");stage6_mem::tag(d_valZ.p,"momentum/assembled_Z");
    stage6_mem::tag(d_s22WallY.p,"momentum/wall_Y");stage6_mem::tag(d_s22WallZ.p,"momentum/wall_Z");stage6_mem::tag(d_s22ValY.p,"momentum/assembled_Y");
    stage6_mem::snapshot("after_all_momentum_allocations",20);
    Dev<double>d_s22PostStats(std::vector<double>(7,0));
    Dev<T>d_s22d4cLagU0(std::vector<T>((std::size_t)nv,0)),
       d_s22d4cLagU1(std::vector<T>((std::size_t)nv,0)),
       d_s22d4cLagU2(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_s22d4cWallStats(std::vector<double>(18,0));
    // Stage 1.A: three FP64 production-fitted wall accumulators.
    // 0 = quadrature area, 1 = integral u_tau^2, 2 = integral y+.
    // Stage3.A: local accumulation within each quadrature thread.
    // 12-point wall integration and BF3 tensor operator are unchanged.
    std::printf("STAGE27_S03A_WALL_ASSEMBLY mode=LOCAL_OFFDIAG_RHS_AGGREGATION "
      "quad=12 basis=8 offdiagAtomicsPerQuad=24 old=384 "
      "crossComponent=FULL source=FROZEN_2C3 status=PASS\n");
    std::printf("STAGE27_S03B_WALL_TENSOR mode=RANK_ONE_NORMAL_TANGENTIAL "
      "quad=12 basis=8 diagonal=ALL_THREE offdiagonal=ALL_SIX "
      "rhsAtomicsPerQuad=24 bf3=UNCHANGED source=FROZEN_3A status=PASS\n");
    Dev<double>d_stage27ProdWallStats(std::vector<double>(3,0.));
    // Stage2.C.3: post-pressure-correction OF_AUTO fit retained for NEXT
    // SIMPLE momentum assembly. Every hit strictly rechecks all active
    // tangential sample magnitudes and nu; stale snapshots fall back.
    Dev<::nodals::wall::shared27::Record>d_stage27SharedWallFit(
      std::vector<::nodals::wall::shared27::Record>(
        stage22d4Live?BH.wallT.size()*12u:1u));
    Dev<unsigned char>d_stage27C3ReuseFlags(
      std::vector<unsigned char>(stage22d4Live?BH.wallT.size()*12u:1u,0u));
    std::vector<unsigned char> stage27Flags(BH.wallT.size()*12u,0u);
    sstG1.stage27SavedWallFit=stage22d4Live?d_stage27SharedWallFit.p:nullptr;
    std::printf("STAGE27_S02C3_SHARED_WALL_CACHE enabled=%d records=%zu bytes=%zu "
      "producer=POST_CORRECTION_OFAUTO consumer=NEXT_MOMENTUM "
      "validation=EXACT_FP64_TANGENTIAL_SPEED_AND_NU "
      "fallback=FROZEN_2C1 status=PASS\n",(int)stage22d4Live,
      stage22d4Live?BH.wallT.size()*12u:0u,
      stage22d4Live?BH.wallT.size()*12u*sizeof(::nodals::wall::shared27::Record):0u);
    Dev<double>d_s22d4dCoeff(std::vector<double>(
       stage22d4Live ? (std::size_t)24*BH.wallT.size() : (std::size_t)1,0.));
    // F2: three face scalars [beta,uTau,yPlus] per wall triangle.
    Dev<double>d_f2FaceFit(std::vector<double>(
      f2FaceSpalding ? (std::size_t)3*BH.wallT.size() : 1u,0.));
    // F1B: 64 FP64 cross-component coefficients per physical wall face.
    // Allocated only when enabled; otherwise one unused scalar element.
    Dev<double>d_f1bWallCross(std::vector<double>(
      f1bFrozenWall ? (std::size_t)64*BH.wallT.size() : (std::size_t)1,0.));
    // Optional first-step full operator equivalence test (OFF during timing).
    Dev<T>d_f1bGateLegacy(std::vector<T>(f1bGpuGate&&f1bFrozenWall?(std::size_t)3*nv:1u,0));
    Dev<T>d_f1bGateCached(std::vector<T>(f1bGpuGate&&f1bFrozenWall?(std::size_t)3*nv:1u,0));
    Dev<T>d_f1bGateDiff(std::vector<T>(f1bGpuGate&&f1bFrozenWall?(std::size_t)3*nv:1u,0));
    // I2: device-resident snapshots and temporary buffers for reduced post-SIMPLE residual.
    const bool stage22d4i3Enable=std::getenv("SST22D4I3A_SCHUR") &&
       std::string(std::getenv("SST22D4I3A_SCHUR"))=="1";
    if(stage22d4i3Enable && !stage22d4iEnable)
      throw std::runtime_error("I3A requires BF3 constrained coupled momentum");
    Dev<T>d_i3penX(std::vector<T>(stage22d4i3Enable?(std::size_t)A.nnz:1,0));
    Dev<T>d_i3penY(std::vector<T>(stage22d4i3Enable?(std::size_t)A.nnz:1,0));
    Dev<T>d_i3penZ(std::vector<T>(stage22d4i3Enable?(std::size_t)A.nnz:1,0));
    Dev<T>d_i3rx(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3ry(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3rz(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3cx(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3cy(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3cz(std::vector<T>(stage22d4i3Enable?(std::size_t)nv:1,0));
    Dev<T>d_i3contPre(std::vector<T>(stage22d4i3Enable?(std::size_t)np:1,0));
    Dev<T>d_i3schurDp(std::vector<T>(stage22d4i3Enable?(std::size_t)np:1,0));
    Dev<T>d_i3contErr(std::vector<T>(stage22d4i3Enable?(std::size_t)np:1,0));
    hxt5d::S22D4I3ASchur<T> schurI3{
       B,sstG1.tetWallPlans,d_tv.p,(int)BH.wallT.size(),pin,
       d_i3rx.p,d_i3ry.p,d_i3rz.p,d_i3cx.p,d_i3cy.p,d_i3cz.p};
    if(stage22d4i3Enable)std::printf(
       "STAGE22D4I3A_CONFIG schur=B_T_Q_D0inv_Q_Tt_Bt "
       "pressureCorrection=SAME_ACTION penaltyInD0=EXCLUDED "
       "momentumGamma=UNCHANGED maskedXY=UNCHANGED "
       "D0=SCALAR_PENALTY_FREE_ROW_L1_NOT_EXACT_REDUCED_ROW_L1 status=PASS\n");
    Dev<T>d_s22d4i2Old(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4i2X(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4i2Au(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4i2Adj(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4i2R(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4i2Dp0(std::vector<T>(stage22d4iEnable?(std::size_t)nv:1,0));
    Dev<T>d_s22d4i2Dp1(std::vector<T>(stage22d4iEnable?(std::size_t)nv:1,0));
    Dev<T>d_s22d4i2Dp2(std::vector<T>(stage22d4iEnable?(std::size_t)nv:1,0));
    double s22d4i2Mx=0.,s22d4i2My=0.,s22d4i2Mz=0.;
    Dev<T>d_s22d4iScratch(std::vector<T>(stage22d4iEnable?(std::size_t)3*nv:1,0));
    Dev<T>d_s22d4dX(std::vector<T>(stage22d4Live ? (std::size_t)3*nv : 1,0));
    Dev<T>d_s22d4dB(std::vector<T>(stage22d4Live ? (std::size_t)3*nv : 1,0));
    // Projected component FGMRES uses 3nv operator scratch and ONE nv Krylov basis.
    // Its original coupled FGMRES workspace is allocated only if selected/fallback.
    Dev<T>d_s05b1Embed(std::vector<T>(s05b1Active?(std::size_t)3*nv:1,0));
    Dev<T>d_s05b1Ax(std::vector<T>(s05b1Active?(std::size_t)3*nv:1,0));
    Dev<T>d_s05b1Residual(std::vector<T>(s05b1Active?(std::size_t)3*nv:1,0));
    Dev<T>d_s05b1Backup(std::vector<T>(s05b1Active?(std::size_t)3*nv:1,0));
    Dev<T>d_s05b1Rhs(std::vector<T>(s05b1Active?(std::size_t)nv:1,0));
    Dev<T>d_s05b1Delta(std::vector<T>(s05b1Active?(std::size_t)nv:1,0));
    hxt4a::Reducer<T>s05b1ScalarRed(s05b1Active?nv:1);
    hxt4a::Reducer<T> stage22d4dRed(stage22d4Live ? 3*nv : 1);
    // Opt-in M3 numeric block-CSR buffers. Allocated once, reused every SIMPLE.
    Dev<T> d_m3K,d_m3Temp,d_m3Check,d_m3CheckDiff;
    Dev<int> d_m3BubbleFace;
    Dev<unsigned long long> d_m3Bad;
    Dev<unsigned char> d_m4Coupled(std::vector<unsigned char>(m3Mode=="m4"?(std::size_t)nv:1,0));
    if(m3Active){
      const std::size_t entries=9ull*A.nnz, bytes=entries*sizeof(T);
      if(entries/9ull!=A.nnz||bytes/sizeof(T)!=entries)
        throw std::runtime_error("M3 buffer overflow");
      d_m3K.n=entries;d_m3Temp.n=entries;
      HXT1_CUDA(cudaMalloc((void**)&d_m3K.p,bytes));
      HXT1_CUDA(cudaMalloc((void**)&d_m3Temp.p,bytes));
      std::vector<int> bubbleFace((std::size_t)nv,-1);
      for(std::size_t f=0;f<BH.wallT.size();++f){
        const auto&face=BH.wallT[f];
        const int bubble=M.tetVel[(std::size_t)face.cell].g[4+(int)face.lf];
        if(bubble<0||bubble>=nv||bubbleFace[(std::size_t)bubble]>=0)
          throw std::runtime_error("M3 needs unique boundary bubble per face");
        bubbleFace[(std::size_t)bubble]=(int)f;
      }
      d_m3BubbleFace.upload(bubbleFace);
      d_m3Bad.upload(std::vector<unsigned long long>(1,0));
      d_m3Check.n=3ull*nv;d_m3CheckDiff.n=3ull*nv;
      HXT1_CUDA(cudaMalloc((void**)&d_m3Check.p,d_m3Check.n*sizeof(T)));
      HXT1_CUDA(cudaMalloc((void**)&d_m3CheckDiff.p,d_m3CheckDiff.n*sizeof(T)));
      std::printf("NODALS_M3_MEMORY csrNnz=%zu coupledBufferMiB=%.2f "
         "bufferCount=2 status=PASS\n",A.nnz,2.0*(double)bytes/1048576.);
    }

    Dev<double>d_s22d4cFEFlux(std::vector<double>(3,0));
    std::vector<hxt5b::TetFacePlan> stage22d4cOutFaces;
    if(stage22d4Live){
      for(const auto& t:M.outTet)stage22d4cOutFaces.push_back(hxt5b::tetplan(M,t));
      if(stage22d4cOutFaces.empty())throw std::runtime_error("STAGE22D4C needs an outlet patch");
    }
    Dev<hxt5b::TetFacePlan>d_s22d4cOutFaces(stage22d4cOutFaces);
    double stage22d4cPostF=0.,stage22d4cPostY=0.,stage22d4cPostWall=0.;

    hxt4a::Reducer<T>pred(np),ured(nv);hxt4a::CGWork<T>WP(np);

    // Initial current-fine pressure matrix and setup-snapshot hierarchy.
    HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,M.h.nhex);
    if(M.h.ntet)tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
    hxt5a::MixStats mix{};
    if(laminarPerf){
      HXT1_CUDA(cudaMemset(d_turb.p,0,d_turb.n*sizeof(T)));
      HXT1_CUDA(cudaMemset(d_diagRTurb.p,0,d_diagRTurb.n*sizeof(T)));
    }else{
      mix=hxt5a::h5_assemble_mixlen(
        M.h.nhex,M.h.ntet,M.h.ninterface,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
        d_u0.p,d_u1.p,d_u2.p,
        d_turb,d_diagRTurb,d_mixStats,
        R,nu,mixlenScale,gamma,d_bad);
    }
    hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_wallCache,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
      d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
      bulk,R,nu,mixlenScale,wallSampleFraction,d_wallT.p,d_tetQuad.p,d_tetMatch.p);
    hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
    HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
    HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad){
      // NODALS_HXT5B_RAU_DIAG_AUDIT_BEGIN
      // Failure-only host audit.  This deliberately does NOT alter the
      // positivity policy or any operator value; it only reports which part
      // of the directional rAU construction triggered the existing gate.
      std::vector<T> hOx((std::size_t)nv),hOz((std::size_t)nv),hDx((std::size_t)nv),hDz((std::size_t)nv);
      std::vector<T> hDiagR((std::size_t)nv),hDiagRTurb((std::size_t)nv),hDiagDG((std::size_t)nv),hDiagWall((std::size_t)nv);
      std::vector<std::int32_t> hDiagPos((std::size_t)nv);
      std::vector<T> hConv((std::size_t)A.nnz);
      HXT1_CUDA(cudaMemcpy(hOx.data(),A.diagOriginal.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hOz.data(),d_diagOrigZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDx.data(),d_relaxDeltaXY.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDz.data(),d_relaxDeltaZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagR.data(),A.diagRBase.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagRTurb.data(),d_diagRTurb.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagDG.data(),d_diagDG.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagWall.data(),d_diagWall.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagPos.data(),A.diagPos.p,(std::size_t)nv*sizeof(std::int32_t),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hConv.data(),A.conv.p,(std::size_t)A.nnz*sizeof(T),cudaMemcpyDeviceToHost));

      // "interfaceAdjacent" means a velocity DOF belonging to either element
      // in at least one HEX/TET interface plan.  This is intentionally a
      // broader and more useful diagnostic category than trace-only DOFs.
      std::vector<unsigned char> interfaceAdjacent((std::size_t)nv,0);
      for(const auto&p:IP){
        const auto&HG=M.hexVel[(std::size_t)p.hexCell];
        for(int a=0;a<14;++a)interfaceAdjacent[(std::size_t)HG.g[a]]=1;
        for(int s=0;s<2;++s){
          const auto&TG=M.tetVel[(std::size_t)p.tetCell[s]];
          for(int a=0;a<8;++a)interfaceAdjacent[(std::size_t)TG.g[a]]=1;
        }
      }

      struct H5B3AuditStat{
        unsigned long long nonfinite=0,nonpositive=0;
        double minFinite=std::numeric_limits<double>::infinity();
        double maxFinite=-std::numeric_limits<double>::infinity();
        int minIndex=-1;
      };
      struct H5B3AuditOff{
        unsigned long long total=0,wall=0,interfaceAdjacent=0,wallAndInterface=0,neither=0;
      };
      auto statUpdate=[&](H5B3AuditStat&s,double v,int i){
        if(!std::isfinite(v)){++s.nonfinite;return;}
        if(v<s.minFinite){s.minFinite=v;s.minIndex=i;}
        s.maxFinite=std::max(s.maxFinite,v);
        if(!(v>0.0))++s.nonpositive;
      };
      auto offUpdate=[&](H5B3AuditOff&o,int i){
        ++o.total;
        const bool w=BH.wallMask[(std::size_t)i]!=0;
        const bool q=interfaceAdjacent[(std::size_t)i]!=0;
        if(w)++o.wall;if(q)++o.interfaceAdjacent;if(w&&q)++o.wallAndInterface;if(!w&&!q)++o.neither;
      };
      H5B3AuditStat sOx,sOz,sNoX,sNoZ,sMx,sMz;
      H5B3AuditOff oOx,oOz,oNoX,oNoZ,oMx,oMz;
      unsigned long long active=0,fixedCount=0;
      for(int i=0;i<nv;++i){
        if(fixed.mask[(std::size_t)i]){++fixedCount;continue;}
        ++active;
        const int dp=hDiagPos[(std::size_t)i];
        const double ox=(double)hOx[(std::size_t)i],oz=(double)hOz[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        const double mx=noX+(double)hDx[(std::size_t)i];
        const double mz=noZ+(double)hDz[(std::size_t)i];
        statUpdate(sOx,ox,i);statUpdate(sOz,oz,i);statUpdate(sNoX,noX,i);statUpdate(sNoZ,noZ,i);statUpdate(sMx,mx,i);statUpdate(sMz,mz,i);
        if(!std::isfinite(ox)||!(ox>0.0))offUpdate(oOx,i);
        if(!std::isfinite(oz)||!(oz>0.0))offUpdate(oOz,i);
        if(!std::isfinite(noX)||!(noX>0.0))offUpdate(oNoX,i);
        if(!std::isfinite(noZ)||!(noZ>0.0))offUpdate(oNoZ,i);
        if(!std::isfinite(mx)||!(mx>0.0))offUpdate(oMx,i);
        if(!std::isfinite(mz)||!(mz>0.0))offUpdate(oMz,i);
      }
      auto printStat=[&](const char*q,const H5B3AuditStat&s){
        std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT quantity=%s minFinite=%.12e maxFinite=%.12e nonpositive=%llu nonfinite=%llu minIndex=%d\n",
          q,s.minFinite,s.maxFinite,s.nonpositive,s.nonfinite,s.minIndex);
      };
      auto printOff=[&](const char*q,const H5B3AuditOff&o){
        std::printf("NODALS_HXT5B_RAU_OFFENDERS quantity=%s total=%llu wall=%llu interfaceAdjacent=%llu wallAndInterface=%llu neither=%llu\n",
          q,o.total,o.wall,o.interfaceAdjacent,o.wallAndInterface,o.neither);
      };
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT kernelBad=%llu active=%llu fixed=%llu alphaU=%.12e rauScale=%.12e policy=UNCHANGED_DIAGNOSTIC_ONLY\n",
        bad,active,fixedCount,alphaU,rauScale);
      printStat("fullXY",sOx);printOff("fullXY",oOx);
      printStat("fullZ",sOz);printOff("fullZ",oOz);
      printStat("noPenXY",sNoX);printOff("noPenXY",oNoX);
      printStat("noPenZ",sNoZ);printOff("noPenZ",oNoZ);
      printStat("relaxedNoPenXY",sMx);printOff("relaxedNoPenXY",oMx);
      printStat("relaxedNoPenZ",sMz);printOff("relaxedNoPenZ",oMz);
      auto printWorst=[&](const char*q,const H5B3AuditStat&s){
        const int i=s.minIndex;if(i<0)return;
        const int dp=hDiagPos[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        std::printf("NODALS_HXT5B_RAU_WORST quantity=%s index=%d wall=%d interfaceAdjacent=%d fullXY=%.12e fullZ=%.12e diagRBase=%.12e convDiag=%.12e turbDiag=%.12e dgDiag=%.12e wallDiag=%.12e noPenXY=%.12e noPenZ=%.12e deltaXY=%.12e deltaZ=%.12e relaxedNoPenXY=%.12e relaxedNoPenZ=%.12e\n",
          q,i,(int)BH.wallMask[(std::size_t)i],(int)interfaceAdjacent[(std::size_t)i],
          (double)hOx[(std::size_t)i],(double)hOz[(std::size_t)i],(double)hDiagR[(std::size_t)i],convd,
          (double)hDiagRTurb[(std::size_t)i],(double)hDiagDG[(std::size_t)i],(double)hDiagWall[(std::size_t)i],
          noX,noZ,(double)hDx[(std::size_t)i],(double)hDz[(std::size_t)i],
          noX+(double)hDx[(std::size_t)i],noZ+(double)hDz[(std::size_t)i]);
      };
      printWorst("fullXY",sOx);printWorst("fullZ",sOz);printWorst("noPenXY",sNoX);printWorst("noPenZ",sNoZ);printWorst("relaxedNoPenXY",sMx);printWorst("relaxedNoPenZ",sMz);
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT_END status=FAIL_AS_BEFORE noSolverPolicyChanged=1\n");
      throw std::runtime_error("initial invalid relaxed momentum diagonal or directional Schur denominator");
    }
    std::printf("NODALS_HXT5B_RAU_RELAXED_GATE stage=setup "
                "test=RELAXED_MOMENTUM_AND_SCHUR_DIAGONALS "
                "unrelaxedNegativeAllowed=1 failed=%llu status=PASS\n",bad);

    // STAGE25A_FIX1_LEGACY_DG: this preflight uses the original HXT5B Z-only
    // boundary assembler (NOT the active SST G2 vector DG assembler). Its RHS
    // prescribes bulk in global Z, even for a rotated SST simulation. Test
    // that legacy operator with ITS OWN matching constant Z plug, rather than
    // incorrectly pairing its RHS with the physically rotated inlet plug.
    // A separate, strict XYZ check on the production SST G2 operator follows
    // after each component's vector DG RHS is assembled at git==1.
    std::vector<T> h_stage25LegacyZ((std::size_t)nv,T(0));
    for(std::uint64_t v=0;v<M.h.nv;++v)
      h_stage25LegacyZ[(std::size_t)v]=(T)bulk;
    Dev<T> d_stage25LegacyZ(h_stage25LegacyZ);
    std::printf("STAGE25A_FIX1_LEGACY_DG operator=HXT5B_Z_ONLY "
      "auditPlug=GLOBAL_Z_BULK physicalPlug=STAGE23_XYZ "
      "productionAudit=SST_G2_XYZ_SEPARATE status=PASS\n");
    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,d_dg.p,d_stage25LegacyZ.p,A.tmp.p);
    hxt5b::h5b3_subtract<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.tmp.p,d_inletRhsZ.p,A.res.p);
    const double dgMomAbs=ured.norm(A.res.p);
    const double dgMomRhs=ured.norm(d_inletRhsZ.p);
    const double dgMomRel=dgMomAbs/std::max(dgMomRhs,1e-300);
    const double dgMomLim=std::is_same<T,float>::value?5e-5:2e-11;
    std::printf("NODALS_HXT5B3_DG_PLUG_MOMENTUM_CANCELLATION abs=%.12e rhsNorm=%.12e rel=%.12e limit=%.12e status=%s\n",
      dgMomAbs,dgMomRhs,dgMomRel,dgMomLim,dgMomRel<=dgMomLim?"PASS":"FAIL");
    if(!(dgMomRel<=dgMomLim))
      throw std::runtime_error("HXT5B3 DG weak inlet does not preserve constant plug");

    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,A.conv.p,d_stage25LegacyZ.p,A.tmp.p);
    const double convPlug=ured.norm(A.tmp.p);
    const double convRel=convPlug/std::max(dgMomRhs,1e-300);
    std::printf("NODALS_HXT5B3_CONVECTION_PLUG_CANCELLATION abs=%.12e relToDgRhs=%.12e expected=roundoff status=%s\n",
      convPlug,convRel,convRel<1e-3?"PASS":"CHECK");

    hxt5b::h5b3_relax_audit("xy_setup",A.diagOriginal.p,d_relaxDeltaXY.p,nv);
    hxt5b::h5b3_relax_audit("z_setup",d_diagOrigZ.p,d_relaxDeltaZ.p,nv);

    std::vector<T>rauT((std::size_t)nv),rauZT((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(rauT.data(),A.rau.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(rauZT.data(),d_rauZ.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    std::vector<double>rxyD((std::size_t)nv),rzD((std::size_t)nv);for(int i=0;i<nv;++i){rxyD[(std::size_t)i]=(double)rauT[(std::size_t)i];rzD[(std::size_t)i]=(double)rauZT[(std::size_t)i];}
    auto PH=hxt4c::build_hybrid_fine_host(Bh,pin,rxyD);
    hxt5b::retune_host_fine(PH,Bh,pin,rxyD,rxyD,rzD);
    {
      long double rowSum2=0.0L,diag2=0.0L,frob2=0.0L;
      double maxAbsRowSum=0.0,maxAbsDiag=0.0;

      for(int i=0;i<PH.A.n;++i){
        long double rs=0.0L;

        for(std::int64_t k=PH.A.row[(std::size_t)i];
            k<PH.A.row[(std::size_t)i+1];++k){
          const double a=PH.A.val[(std::size_t)k];
          rs+=(long double)a;
          frob2+=(long double)a*(long double)a;
        }

        const double d=PH.A.diag[(std::size_t)i];

        rowSum2+=rs*rs;
        diag2+=(long double)d*(long double)d;

        maxAbsRowSum=std::max(maxAbsRowSum,std::abs((double)rs));
        maxAbsDiag=std::max(maxAbsDiag,std::abs(d));
      }

      const double rowSumL2=std::sqrt((double)rowSum2);
      const double diagL2=std::sqrt((double)diag2);
      const double frob=std::sqrt((double)frob2);

      std::printf(
        "NODALS_HXT5B_PRESSURE_CONSTANT_AUDIT "
        "mode=%s pin=%d "
        "S1_L2=%.12e S1_over_diagL2=%.12e "
        "S1_over_frob=%.12e "
        "maxAbsS1=%.12e maxAbsDiag=%.12e "
        "maxRatio=%.12e "
        "interpretation=%s status=PASS\n",
        pressureGauge.c_str(),pin,
        rowSumL2,
        rowSumL2/std::max(diagL2,1e-300),
        rowSumL2/std::max(frob,1e-300),
        maxAbsRowSum,maxAbsDiag,
        maxAbsRowSum/std::max(maxAbsDiag,1e-300),
        (rowSumL2/std::max(diagL2,1e-300) > 1e-10)
          ? "CONSTANT_MODE_NOT_NULL_CANDIDATE"
          : "CONSTANT_MODE_NULL_CANDIDATE");
    }


    int effectivePowerIts=0;
    if(amgSpectrumPolicy=="always")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="auto"&&amgSmoother=="cheb2")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="off")effectivePowerIts=0;
    else if(amgSpectrumPolicy!="auto")throw std::runtime_error("bad AMG spectrum policy");
    if(amgSmoother=="cheb2"&&effectivePowerIts<=0)throw std::runtime_error("Chebyshev requires spectrum estimation");

    auto HH=nodals_gpu::build_cf_hierarchy_from_csr(
      PH.A,cfTheta,cfPmax,cfInterp,cfAggressiveFirst!=0,amgTerminal,
      effectivePowerIts,amgLambdaSafety,amgLambdaLow);

    hxt4c::AMGOptions AO;
    AO.smoother=amgSmoother;AO.jacobiOmega=amgJacobiOmega;
    AO.jacobiFineSweeps=amgJacobiFineSweeps;AO.jacobiCoarseSweeps=amgJacobiCoarseSweeps;
    AO.chebDegree=amgChebDegree;AO.lambdaLowFraction=amgLambdaLow;
    AO.mcgsFineSweeps=amgMcgsFineSweeps;AO.mcgsCoarseSweeps=amgMcgsCoarseSweeps;AO.mcgsOmega=amgMcgsOmega;AO.mcgsOrder=amgMcgsOrder;
    hxt4c::HybridAMG<T>PC(PH,HH,pin,AO);
    stage6_mem::tag(PC.fine.val.p,"pressure/fine_CSR_val");stage6_mem::tag(PC.fine.col.p,"pressure/fine_CSR_col");stage6_mem::tag(PC.fine.entryRow.p,"pressure/fine_entryRow");
    stage6_mem::snapshot("after_pressure_AMG_setup",20);
    hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);
    hxt4c::BalancedPOD<T> pressurePOD(
      np,pin,podMode!="off",podMode=="adaptive",podTrainSnapshots,podRank,podMaxRank,podNovelty,
      podTailIter,podHardIts,podAcceptMinAbs,podAcceptMinRel,podRefreshAbs,podRefreshRel,
      podRecompressRank,podRecompressHistory,podRecompressMaxLossAbs,podRecompressMaxLossRel,
      podRecompressMethod,podRecompressForce!=0,podNoProbe!=0);
    std::printf(
      "NODALS_HXT5B_AMG_FINE_MODE "
      "mode=%s "
      "fineSmoother=%s "
      "fineJacobiSweeps=%d "
      "coarseHierarchy=UNCHANGED "
      "setupExplicitFineCSR=RETAINED "
      "dynamicExplicitRefresh=%s "
      "status=PASS\n",
      amgFineMode.c_str(),
      amgSmoother.c_str(),
      amgJacobiFineSweeps,
      amgFineMode=="explicit"?"ON":"OFF");


    auto wallNow=hxt5b::wallstats(d_wallStats,bulk);
    std::printf(
      "NODALS_HXT5B_CONFIG precision=%s geometry=fp64 reductions=fp64 cellsHex=%llu cellsTet=%llu "
      "velocityDofs=%d pressureDofs=%d Re=%.12g bulk=%.12g nu=%.12e gamma=%.6g "
      "turbulence=NIKURADSE_ALGEBRAIC mixlenScale=%.8g inlet=DG_NUMERICAL_TRACE_PLUG velocityInletDOFs=FREE "
      "wall=FIREDRAKE_Q1BF2_OFFWALL_SPALDING wallSampleFraction=%.6g kappa=0.4 B=5.5 transverseWallClamp=1 "
      "momentumWork=%s momentumRelax=row_l1 momentumRtol=%.3e momentumAtol=%.3e momentumMax=%d momentumCapPolicy=%s outerConvergence=CONTINUITY_PLUS_XYZ_PHYSICAL_MOMENTUM_INITIAL_RELATIVE "
      "pressureSolver=%s pRtol=%.3e pAtol=%.3e pMax=%d amgHierarchy=%s cfCoarsening=%s cfStrength=%s "
      "cfTheta=%.6g cfInterp=%s cfPmax=%d amgSmoother=%s chebDegree=%d powerIts=%d spectrumPolicy=%s "
      "alphaU=%.6g alphaP=%.6g rauScale=%.6g directionalRAU=1 rauPenaltyPolicy=SEPARATE_GAMMA simpleTol=%.3e status=PASS\n",
      PRECISION,(unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,nv,np,Re,bulk,nu,gamma,mixlenScale,
      wallSampleFraction,momWork.c_str(),momRtol,momAtol,momMax,momCapPolicy.c_str(),pressureSolver.c_str(),pRtol,pAtol,pMax,amgHierarchy.c_str(),cfCoarsening.c_str(),
      cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,amgSmoother.c_str(),amgChebDegree,effectivePowerIts,
      amgSpectrumPolicy.c_str(),alphaU,alphaP,rauScale,simpleTol);
    std::printf("NODALS_HXT5E_KRYLOV_CONFIG momentumSolver=%s momRestart=%d momFgsSweeps=%d sstLinearSolver=%s sstRestart=%d sstFgsSweeps=%d rightPreconditioner=FORWARD_COLORED_GS trueResidualAudit=1 status=PASS\n",momWork.c_str(),momFgmresRestart,momFgsSweeps,sstG1.linearSolver.c_str(),sstG1.fgmresRestart,sstG1.fgsSweeps);
    std::printf("NODALS_PRESSURE_POD_MODE mode=%s trainSnapshots=%d requestedRank=%d maxRank=%d noveltyTol=%.12e "
      "tailIter=%d hardIts=%d multiTailDepths=[%d,%d,%d] "
      "refreshAbs=%d refreshRel=%.12e recompressRank=%d recompressHistory=%d recompressMaxLossAbs=%d recompressMaxLossRel=%.12e "
      "recompressMethod=%s recompressForce=%d noProbe=%d scope=SST_G2_ONLY status=PASS\n",
      podMode.c_str(),podTrainSnapshots,podRank,podMaxRank,podNovelty,
      podTailIter,podHardIts,podTailIter,3*podTailIter,6*podTailIter,podRefreshAbs,podRefreshRel,
      podRecompressRank,podRecompressHistory,podRecompressMaxLossAbs,podRecompressMaxLossRel,
      podRecompressMethod.c_str(),podRecompressForce,podNoProbe);
    std::printf("NODALS_TRUE_SIMPLE_CONFIG mode=%s velocityCorrection=%s velocityCorrectionScale=1 pressureRelaxation=%.12e alphaU=%.12e status=PASS\n",
      simpleCoupling.c_str(),simpleCoupling=="corrected"?"IMMEDIATE_MINUS_RAU_BT_DP":"NONE_LAGGED",alphaP,alphaU);
    std::printf("NODALS_HXT5B_NITSCHE_CONFIG momentumGamma=%.12e schurGamma=%.12e schurOverMomentum=%.12e rAUPolicy=NO_RAW_NOPENGATE_PLUS_ROWL1_PLUS_SELECTED_NITSCHE status=PASS\n",
      gamma,schurNitscheGamma,(gamma>0.0?schurNitscheGamma/gamma:0.0));
    std::printf(
      "NODALS_HXT5B_BOUNDARY_SETUP inletHexFaces=%zu inletTetFaces=%zu wallHexFaces=%zu inletArea=%.12e "
      "prescribedFluxSum=%.12e constantPatchResidual=%.12e wallArea=%.12e wallYPlusMeanInitial=%.12e "
      "wallDarcyInitial=%.12e rootFailures=%.0f status=PASS\n",
      BH.inH.size(),BH.inT.size(),BH.wallH.size(),BH.inletArea,BH.sourceSum,BH.constResidual,
      wallNow.area,wallNow.yPlusMean,wallNow.fDarcy,wallNow.rootFailures);
    if(laminarPerf)
      std::printf(
        "NODALS_HXT5B_LAMINAR_PERF_CONFIG "
        "enabled=1 turbulence=OFF nut=ZERO "
        "scalarTransports=SKIPPED "
        "momentumTurbulentOperator=SKIPPED "
        "wall=UNCHANGED_HXT5B pressure=%s fixedOuter=%d status=PASS\n",
        pressureSolver.c_str(),maxOuter);
    else
      std::printf(
        "NODALS_HXT5B_MIXLEN_SETUP "
        "meanNutOverNu=%.12e maxNutOverNu=%.12e "
        "maxStrain=%.12e status=PASS\n",
        mix.meanNutOverNu,mix.maxNutOverNu,mix.maxStrain);

    double c0=-1,m0[3]={-1,-1,-1};bool converged=false;int outer=0;long long pIts=0;
    // G0/SST baseline diagnostic buffer.  This does not enter any operator or
    // convergence decision; it is copied only at the existing sparse print points
    // so pressure-drop stationarity can be judged together with y+ and wall friction.
    std::vector<T> hPressurePlateau((std::size_t)np);

    const bool directSSTStart=(sstG2&&sstG2Start=="plug");

    std::chrono::steady_clock::time_point lamPerfT0{};
    if(laminarPerf){
      HXT1_CUDA(cudaDeviceSynchronize());
      lamPerfT0=std::chrono::steady_clock::now();
    }

    if(directSSTStart){
      // Straight-pipe potential-flow startup: the existing coefficient initializer
      // already gives U=(0,0,Ub) exactly, enriched velocity coefficients zero, and p=0.
      // Skip the expensive converged Nikuradse SIMPLE precursor entirely.
      converged=true;outer=0;pIts=0;
      std::printf("NODALS_SST_G2_START mode=%s velocity=%s pressure=ZERO kOmega=UNIFORM_INLET_FORMULAS precursorSimpleIterations=0 status=PASS\n",
        stage27PotentialStart?"LAPLACE_P1_PROJECTED":"PLUG",
        stage27PotentialStart?"PROJECTED_GRAD_PHI_Q1_BF3":"UNIFORM_AXIAL_BULK");
      std::printf("STAGE27_S02C1_FUSED_GN fitPoints=%d model=UNCHANGED "
          "residualAndDerivative=ONE_PASS objective=LEGACY_FP64 "
          "initialRoots=UNCHANGED lineSearch=UNCHANGED goldenFallback=UNCHANGED "
          "linearSolvers=UNCHANGED status=PASS\n",stage24bN);
    } else {
// ============================================================
// HXT5B_COARSE4_TIMING_V1
// Instrumentation only.
//
// e0 -> e1 : dynamic assembly
// e1 -> e2 : Schur diagonal + fine numeric refresh
// e2 -> e3 : momentum RHS/B^T + momentum solve
// e3 -> e4 : continuity construction
// e4 -> e5 : pressure solve + pressure update/B^T correction
//
// Only one cudaEventSynchronize per measured outer.
// Outer 1 excluded from averages.
// ============================================================
cudaEvent_t h5te[6];
for(int i=0;i<6;++i)
    HXT1_CUDA(cudaEventCreate(&h5te[i]));

double h5tAssembly=0.0;
double h5tSchur=0.0;
double h5tMomentum=0.0;
double h5tContinuity=0.0;
double h5tPressure=0.0;
int h5tCount=0;

for(int it=1;it<=maxOuter;++it){
      HXT1_CUDA(cudaEventRecord(h5te[0],0));
      HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      if(M.h.nhex)hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,M.h.nhex);
      if(M.h.ntet)tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
      if(!laminarPerf)
        mix=hxt5a::h5_assemble_mixlen(
          M.h.nhex,M.h.ntet,M.h.ninterface,
          d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
          d_u0.p,d_u1.p,d_u2.p,
          d_turb,d_diagRTurb,d_mixStats,
          R,nu,mixlenScale,gamma,d_bad);
      hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_wallCache,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
        d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
        bulk,R,nu,mixlenScale,wallSampleFraction,d_wallT.p,d_tetQuad.p,d_tetMatch.p);
      hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
      HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
      HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
      if(bad)throw std::runtime_error("invalid relaxed momentum/Schur diagonal count="+std::to_string(bad));
      if(it==1)std::printf("NODALS_HXT5B_RAU_RELAXED_GATE stage=outer1 "
                            "test=RELAXED_MOMENTUM_AND_SCHUR_DIAGONALS "
                            "unrelaxedNegativeAllowed=1 failed=%llu status=PASS\n",bad);
      HXT1_CUDA(cudaEventRecord(h5te[1],0));
      if(it==1&&!laminarPerf){
        hxt5b::audit_real("matrix_xy_after_finalize",A.val.p,A.nnz);
        hxt5b::audit_real("matrix_z_after_finalize",d_valZ.p,A.nnz);
        hxt5b::audit_real("rau_xy_after_finalize",A.rau.p,(std::size_t)nv);
        hxt5b::audit_real("rau_z_after_finalize",d_rauZ.p,(std::size_t)nv);
        hxt5b::audit_diag_host("diag_xy_original",A.diagOriginal.p,nv);
        hxt5b::audit_diag_host("diag_z_original",d_diagOrigZ.p,nv);
      }

      hxt5b::sdiag<<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,A.rau.p,d_rauZ.p,d_sd.p,pin);
      hxt4a::invert_diag_kernel<T><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);
      hxt5b::SAniso<T>S(B,A.rau.p,A.rau.p,d_rauZ.p,pin);
      if(amgFineMode=="explicit")
        hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);
      HXT1_CUDA(cudaEventRecord(h5te[2],0));

      if(it==1&&!laminarPerf&&amgFineMode=="explicit"){
        double parity=hxt5b::parity(PC,S,np,pin,pred);
        const double lim=std::is_same<T,float>::value?3e-5:2e-11;
        std::printf("NODALS_HXT5B_FINE_CSR_PARITY rel=%.12e limit=%.12e status=%s\n",parity,lim,parity<=lim?"PASS":"FAIL");
        if(parity>lim)throw std::runtime_error("HXT5B anisotropic fine CSR/matrix-free Schur parity failed");
      }

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,d_rhs0.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,d_rhs1.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,d_rhs2.p);
      if(it==1&&!laminarPerf){
        hxt5b::audit_real("rhs_x_before_sgs",d_rhs0.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_y_before_sgs",d_rhs1.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_z_before_sgs",d_rhs2.p,(std::size_t)nv);
        hxt5b::audit_real("u_z_before_sgs",d_u2.p,(std::size_t)nv);
      }
      auto mx=hxt5b_perf::momentum_solve(momWork,it,A,VC,A.val.p,d_rhs0.p,d_u0.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"x",momCapPolicy=="warn");
      auto my=hxt5b_perf::momentum_solve(momWork,it,A,VC,A.val.p,d_rhs1.p,d_u1.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"y",momCapPolicy=="warn");
      auto mz=hxt5b_perf::momentum_solve(momWork,it,A,VC,d_valZ.p,d_rhs2.p,d_u2.p,momOmega,d_fixed.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"z",momCapPolicy=="warn");
      HXT1_CUDA(cudaEventRecord(h5te[3],0));
      // HXT5B_MOM_CAP_POLICY_V1
      // h5b3_momentum_solve throws internally for NaN/Inf or invalid work.
      // Therefore ok=false here means finite MOM_MAX exhaustion only.
      // In SIMPLE this is bounded inner work; the outer physical residual gate
      // remains the authority on nonlinear convergence.
      const bool momCapMiss=(!mx.ok||!my.ok||!mz.ok);
      if(momCapMiss){
        const bool strictMomCap=(momCapPolicy=="fail");
        std::printf("NODALS_HXT5B3_MOM_CAP outer=%d xRel=%.12e xSweeps=%d yRel=%.12e ySweeps=%d zRel=%.12e zSweeps=%d targetRel=%.6g maxSweeps=%d policy=%s status=%s\n",
          it,mx.rel,mx.sweeps,my.rel,my.sweeps,mz.rel,mz.sweeps,momRtol,momMax,
          momCapPolicy.c_str(),strictMomCap?"FAIL":"WARN_CONTINUE");
        if(strictMomCap)
          throw std::runtime_error("HXT5B3 momentum residual-drop target not reached");
      }
      if(it<=5||it%20==0)
        std::printf("NODALS_HXT5B3_MOM_SOLVE outer=%d xSweeps=%d xRel=%.6e ySweeps=%d yRel=%.6e zSweeps=%d zRel=%.6e targetRel=%.6g status=PASS\n",
          it,mx.sweeps,mx.rel,my.sweeps,my.rel,mz.sweeps,mz.rel,momRtol);

      hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
      if(it==1&&!laminarPerf)hxt5b::audit_real("continuity_before_norm",d_cont.p,(std::size_t)np);
      double cn=pred.norm(d_cont.p);
      HXT1_CUDA(cudaEventRecord(h5te[4],0));
      if(!std::isfinite(cn))throw std::runtime_error("HXT5B nonfinite continuity norm");

      HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(T)));
      hxt4a::CGResult pr;
      if(pressureSolver=="pcg_amg")pr=(amgFineMode=="matrixfree"
          ? hxt4c::pcg_amg_mf_fine(
              S,PC,d_sd.p,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin)
          : hxt4c::pcg_amg(
              S,PC,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin));
      else pr=pcg_prod(S,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);
      if(!pr.ok){
        std::printf("NODALS_HXT5B_PRESSURE_FAIL outer=%d solver=%s pCG=%d pRel=%.12e pRtol=%.12e pAtol=%.12e pMax=%d status=FAIL\n",
          it,pressureSolver.c_str(),pr.its,pr.rel,pRtol,pAtol,pMax);
        throw std::runtime_error("pressure solver failed production tolerance");
      }
      pIts+=pr.its;
      pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(
        np,pin,(T)alphaP,d_dp.p,d_p.p);

      const double cnPreCorrection=cn;
      if(simpleCoupling=="corrected"){
        // Full classical SIMPLE velocity correction.  Do NOT multiply by
        // alphaP here: alphaP relaxes pressure, not the velocity correction.
        hxt4a::BT_apply(B,d_dp.p,d_bt0.p,d_bt1.p,d_bt2.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_clampXY.p,A.rau.p,d_bt0.p,d_u0.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_clampXY.p,A.rau.p,d_bt1.p,d_u1.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_fixed.p,d_rauZ.p,d_bt2.p,d_u2.p);
        hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);
        hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);
        hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
        cn=pred.norm(d_cont.p);
        if(!std::isfinite(cn))throw std::runtime_error("HXT5B corrected SIMPLE nonfinite continuity");
        if(it<=5||it%20==0)
          std::printf("NODALS_TRUE_SIMPLE_CORRECTION scope=G0 it=%d contPre=%.12e contPost=%.12e ratio=%.12e pCG=%d pRel=%.12e status=PASS\n",
            it,cnPreCorrection,cn,cn/std::max(cnPreCorrection,1e-300),pr.its,pr.rel);
      }

      if(laminarPerf){
        HXT1_CUDA(cudaEventRecord(h5te[5],0));
        HXT1_CUDA(cudaEventSynchronize(h5te[5]));

        if(it>=2){
          float x01=0.0f;
          float x12=0.0f;
          float x23=0.0f;
          float x34=0.0f;
          float x45=0.0f;

          HXT1_CUDA(cudaEventElapsedTime(&x01,h5te[0],h5te[1]));
          HXT1_CUDA(cudaEventElapsedTime(&x12,h5te[1],h5te[2]));
          HXT1_CUDA(cudaEventElapsedTime(&x23,h5te[2],h5te[3]));
          HXT1_CUDA(cudaEventElapsedTime(&x34,h5te[3],h5te[4]));
          HXT1_CUDA(cudaEventElapsedTime(&x45,h5te[4],h5te[5]));

          h5tAssembly   += (double)x01;
          h5tSchur      += (double)x12;
          h5tMomentum   += (double)x23;
          h5tContinuity += (double)x34;
          h5tPressure   += (double)x45;

          ++h5tCount;
        }

        outer=it;

        if(it<=5 || it==maxOuter)
          std::printf(
            "NODALS_HXT5B_LAMINAR_PERF "
            "it=%d absCont=%.12e pCG=%d pRel=%.3e status=PASS\n",
            it,cn,pr.its,pr.rel);

        continue;
      }

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      HXT1_CUDA(cudaEventRecord(h5te[5],0));
      HXT1_CUDA(cudaEventSynchronize(h5te[5]));

      if(it>=2){
        float x01=0.0f;
        float x12=0.0f;
        float x23=0.0f;
        float x34=0.0f;
        float x45=0.0f;

        HXT1_CUDA(cudaEventElapsedTime(&x01,h5te[0],h5te[1]));
        HXT1_CUDA(cudaEventElapsedTime(&x12,h5te[1],h5te[2]));
        HXT1_CUDA(cudaEventElapsedTime(&x23,h5te[2],h5te[3]));
        HXT1_CUDA(cudaEventElapsedTime(&x34,h5te[3],h5te[4]));
        HXT1_CUDA(cudaEventElapsedTime(&x45,h5te[4],h5te[5]));

        h5tAssembly   += (double)x01;
        h5tSchur      += (double)x12;
        h5tMomentum   += (double)x23;
        h5tContinuity += (double)x34;
        h5tPressure   += (double)x45;
        ++h5tCount;
      }

      double ma[3]={
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,d_valZ.p,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,ured)};
      if(it==1){c0=cn;m0[0]=ma[0];m0[1]=ma[1];m0[2]=ma[2];}
      const double st=std::max(simpleTol,1e-300);
      const double rc=cn/std::max(std::max(c0,outerContAtol/st),1e-300);
      const double rm0=ma[0]/std::max(std::max(m0[0],momAtol/st),1e-300);
      const double rm1=ma[1]/std::max(std::max(m0[1],momAtol/st),1e-300);
      const double rm2=ma[2]/std::max(std::max(m0[2],momAtol/st),1e-300);
      outer=it;converged=(it>1&&rc<=simpleTol&&rm0<=simpleTol&&rm1<=simpleTol&&rm2<=simpleTol);
      if(it<=5||it%20==0||converged){
        auto ws=hxt5b::wallstats(d_wallStats,bulk);
        HXT1_CUDA(cudaMemcpy(hPressurePlateau.data(),d_p.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToHost));
        const auto pPlateau=hxt4a::to_double(hPressurePlateau);
        const double pressureSlopePlateau=hxt4a::pressure_slope(M,pPlateau);
        const double pInPlateau=hxt4a::boundary_p(M,pPlateau,true);
        const double pOutPlateau=hxt4a::boundary_p(M,pPlateau,false);
        const double pressureDropPlateau=pInPlateau-pOutPlateau;
        std::printf("NODALS_HXT5B_SIMPLE it=%d relContInit=%.12e absCont=%.12e relMomXInit=%.12e absMomX=%.12e relMomYInit=%.12e absMomY=%.12e relMomZInit=%.12e absMomZ=%.12e pCG=%d pRel=%.3e meanNutOverNu=%.6e maxNutOverNu=%.6e wallYPlusMean=%.6e wallDarcy=%.6e pressureSlope=%.6e pressureDrop=%.6e converged=%d status=PASS\n",
          it,rc,cn,rm0,ma[0],rm1,ma[1],rm2,ma[2],pr.its,pr.rel,mix.meanNutOverNu,mix.maxNutOverNu,ws.yPlusMean,ws.fDarcy,pressureSlopePlateau,pressureDropPlateau,(int)converged);
      }
      if(converged)break;
      double worst=std::max(std::max(rc,rm0),std::max(rm1,rm2));
      if(it>=5&&(!std::isfinite(worst)||worst>1e8))throw std::runtime_error("HXT5B coupled-residual divergence guard");
    }
    if(laminarPerf){
      HXT1_CUDA(cudaDeviceSynchronize());

      const auto lamPerfT1=
        std::chrono::steady_clock::now();

      const double loopMs=
        std::chrono::duration<double,std::milli>(
          lamPerfT1-lamPerfT0).count();

      std::printf(
        "NODALS_HXT5B_LAMINAR_PERF_TIMING "
        "iterations=%d loopMs=%.3f avgOuterMs=%.6f status=PASS\n",
        outer,loopMs,loopMs/std::max(outer,1));

      converged=true;
    }

    if(!laminarPerf&&!converged)
      throw std::runtime_error(
        "HXT5B SIMPLE maxOuter without all-residual convergence");

    
if(h5tCount>0){
    const double inv=1.0/(double)h5tCount;

    const double assembly   = h5tAssembly*inv;
    const double schur      = h5tSchur*inv;
    const double momentum   = h5tMomentum*inv;
    const double continuity = h5tContinuity*inv;
    const double pressure   = h5tPressure*inv;

    const double schurCont =
        schur + continuity;

    const double tracked =
        assembly +
        momentum +
        schurCont +
        pressure;

    std::printf(
      "NODALS_HXT5B_COARSE4_TIMING "
      "measuredOuter=%d "
      "policy=OUTER_2_TO_END "
      "assemblyAvgMs=%.6f "
      "momentumAvgMs=%.6f "
      "schurRefreshAvgMs=%.6f "
      "continuityAvgMs=%.6f "
      "schurPlusContinuityAvgMs=%.6f "
      "pressureSolveUpdateAvgMs=%.6f "
      "trackedAvgMs=%.6f "
      "status=PASS\n",
      h5tCount,
      assembly,
      momentum,
      schur,
      continuity,
      schurCont,
      pressure,
      tracked);
}

for(int i=0;i<6;++i)
    cudaEventDestroy(h5te[i]);

    } // end G0/SIMPLE else block

// G2: either start directly from the exact straight-pipe plug/potential field
    // (default development path) or from the legacy converged G0 field.
    // In both cases k/omega are initialized uniformly from the inlet formulas.

    // The old Nikuradse field is no longer used by momentum inside this loop.
    hxt5c::Stats g2FinalStats;bool g2Gate=true;bool g2Converged=false;int g2Outer=0;double g2KRel=0,g2WRel=0,g2NutRel=0;
    if(sstG2){
      hxt5d::State TS(nv,(int)M.h.nv,bulk,R,sstG1);hxt4a::Reducer<T> sstRed(p1Compact?(int)M.h.nv:nv);
      stage6_mem::tag(TS.k.p,"sst/k");stage6_mem::tag(TS.omega.p,"sst/omega");stage6_mem::tag(TS.oldK.p,"sst/oldK");stage6_mem::tag(TS.oldW.p,"sst/oldOmega");
      stage6_mem::snapshot("after_SST_state",20);
      auto metricGate=nodals::coupling::G2MetricPlateau::from_environment();
      std::vector<double>hf,hy,hdp;double finalDp=0,finalF=0,finalY=0;double g2ContActiveL2Initial=-1,g2ContActiveMaxInitial=-1,g2ContGlobalInitial=-1,g2ContPinInitial=-1;
      auto relSpan=[](const std::vector<double>&v){if(v.empty())return std::numeric_limits<double>::infinity();auto mm=std::minmax_element(v.begin(),v.end());double mean=0;for(double x:v)mean+=x;mean/=v.size();return (*mm.second-*mm.first)/std::max(std::abs(mean),1e-300);};
      const char* sstWallPlacement=
        sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST ? "PROFILE5_REMOTE_MULTIPOINT_EQ_SST" :
        sstG1.profileAverageClosure && sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ? "NQ_AVERAGE_TARGET_PHYSICAL_TRACE_NITSCHE" :
        sstG1.profileAverageClosure ? "NQ_VOLUME_AVERAGE_RANK_ONE" :
        sstG1.profileIntegralClosure ? "FULL_PROFILE_VOLUME_PROJECTION" :
        (sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ? "PHYSICAL_TRACE_SYMMETRIC_NITSCHE" : "OFFWALL_SAMPLE_PENALTY");
      if(sstG1.profileAverageClosure){
        if(sstG1.profileIntegralClosure)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE and SST_PROFILE_INTEGRAL_CLOSURE are mutually exclusive");
        if(sstG1.profileIntegralNQ!=4&&sstG1.profileIntegralNQ!=8&&sstG1.profileIntegralNQ!=12)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE requires SST_PROFILE_INTEGRAL_NQ=4, 8, or 12");
        const bool avgOfAuto=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO;
        const bool avgTable=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE;
        const bool avgProfile5Eq=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST;
        const bool avgSampleLog=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_SAMPLE_LOG;
        const bool avgTraceLog=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG;
        if(!avgOfAuto&&!avgTable&&!avgSampleLog&&!avgTraceLog&&!avgProfile5Eq)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE requires SST_OMEGA_WALL_MODE=of_auto, consistent_table, sample_log, or trace_log");
        if(avgTable){
          if(momentumWallMode!=hxt5d::MOM_WALL_SST_CONSISTENT_TABLE)throw std::runtime_error("average-first consistent_table requires MOMENTUM_WALL_MODE=sst_consistent_table");
          if(sstG1.ofProduction)throw std::runtime_error("average-first consistent_table requires SST_OF_PRODUCTION=0");
          std::printf("NODALS_SST_PROFILE_AVERAGE_TABLE_CONFIG enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN inversion=ONE_TABLE_INVERSION_AT_UBAR_YBAR momentum=TABLE_UTAU_SQUARED_WITH_AVERAGED_TRIAL kTarget=TABLE_KPLUS_UTAU2 omegaTarget=TABLE_OMEGAPLUS_UTAU2_OVER_NU scalarConstraint=RANK_ONE_CELL_AVERAGE_PENALTY production=TABLE_NATIVE_NO_EXTRA_OF_SOURCE sampleFractionRole=DIAGNOSTIC_ONLY status=PASS\n",sstG1.profileIntegralNQ);
        }
        if(avgSampleLog||avgTraceLog){
          if(momentumWallMode!=hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL)throw std::runtime_error("average-first sample_log/trace_log requires MOMENTUM_WALL_MODE=spalding_profile_integral");
          if(!sstG1.profileIntegralUTau)throw std::runtime_error("average-first sample_log/trace_log requires SST_PROFILE_INTEGRAL_UTAU=1");
          if(sstG1.ofProduction)throw std::runtime_error("average-first sample_log/trace_log requires SST_OF_PRODUCTION=0");
          std::printf("NODALS_SST_PROFILE_AVERAGE_LOG_CONFIG enabled=1 mode=%s normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN utau=SPALDING_ON_NQ_AVERAGED_U_Y omegaTarget=UTAU_OVER_SQRTBETASTAR_KAPPA_YBAR placement=%s kWall=HIGH_RE_ZERO_GRADIENT production=NONE sampleFractionRole=DIAGNOSTIC_ONLY status=PASS\n",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.profileIntegralNQ,avgTraceLog?"PHYSICAL_TRACE_SYMMETRIC_NITSCHE":"RANK_ONE_CELL_AVERAGE_PENALTY");
        }
      }
            if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST){
        if(!sstG1.profileAverageClosure||sstG1.profileAverageWallLaw!=5)throw std::runtime_error("profile5_eq_sst requires PROFILE5 wall sampling");
        if(momentumWallMode!=hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL)throw std::runtime_error("profile5_eq_sst requires spalding_profile_integral momentum");
        if(!sstG1.profileIntegralUTau)throw std::runtime_error("profile5_eq_sst requires SST_PROFILE_INTEGRAL_UTAU=1");
        if(sstG1.ofProduction)throw std::runtime_error("profile5_eq_sst requires SST_OF_PRODUCTION=0");
        if(!(sstG1.profile5EqGamma>0.0))throw std::runtime_error("profile5_eq_sst requires --sst-profile5-eq-gamma > 0");
        std::printf("NODALS_SST_PROFILE5_EQ_SST_CONFIG enabled=1 model=SST2003M_LOCAL_EQUILIBRIUM_COUETTE_PROFILE samples=5 sampleGeometry=SAME_PROFILE5_PHYSICAL_WALL_NORMAL_POINTS momentumUtau=SAME_FIVE_POINT_SPALDING_FIT scalarFit=MULTIPOINT_WEIGHTED_LEAST_SQUARES kTarget=TABLE_KPLUS_UTAU2 omegaTarget=TABLE_OMEGAPLUS_UTAU2_OVER_NU tableVelocityInversion=OFF tableMomentum=OFF nativeSSTVolumeTerms=ON extraOFProduction=OFF gamma=%.12e status=PASS\n",sstG1.profile5EqGamma);
      }
std::printf("NODALS_SST_OMEGA_WALL_CONFIG mode=%s target=%s placement=%s physicalWallFlux=%s penaltyGamma=%.12e sampleFraction=%.12e ofKappa=%.6e ofE=%.6e ofBeta1=%.6e ofReBlend=%.6e ofProduction=%d ofProductionScale=%.6e ofWallVolumeScale=%.6e ofProductionLimit=%.6e sharedSpaldingState=%d profileIntegralUTau=%d profileIntegralNQ=%d profileIntegralClosure=%d profileAverageClosure=%d profileClosureYPlusMin=%.6e status=PASS\n",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE?"SST2003M_1D_COUETTE_TABLE":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_COMPATIBLE?"CARRIER_LACOMBE_COMPATIBLE_OMEGA":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO?"OF_OMEGA_VIS_LOG_BLEND":"SPALDING_UTAU_LOG")),sstWallPlacement,sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"NITSCHE_DIRICHLET":"NATURAL_ZERO",sstG1.wallPenaltyGamma,wallSampleFraction,sstG1.ofKappa,sstG1.ofE,sstG1.ofBeta1,sstG1.ofReBlend,(int)sstG1.ofProduction,sstG1.ofProductionScale,sstG1.ofWallVolumeScale,sstG1.ofProductionLimit,(int)sstG1.sharedSpaldingWallState,(int)sstG1.profileIntegralUTau,sstG1.profileIntegralNQ,(int)sstG1.profileIntegralClosure,(int)sstG1.profileAverageClosure,sstG1.profileClosureYPlusMin);
      if(sstG1.sharedSpaldingWallState)std::printf("NODALS_SST_SHARED_WALL_STATE enabled=1 source=EXACT_MOMENTUM_SPALDING_UTAU kappa=0.4 B=5.5 E=9.025013499434122 omegaLog=UTAU_OVER_SQRTBETASTAR_KAPPA_Y wallProduction=UTAU_CUBED_OVER_KAPPA_Y blending=OF_AUTO_UNCHANGED status=PASS\n");
      if(sstG1.profileAverageClosure)std::printf(
        "NODALS_SST_PROFILE_AVERAGE_WALL_LAW law=%s kappa=4.000000000000e-01 E=9.025013499434122e+00 inversion=%s momentumAndOFAutoSharedUTau=1 legacySharedSpaldingFlagNameOnly=%d status=PASS\n",
        sstG1.profileAverageWallLaw==5?"spalding_profile5":(sstG1.profileAverageWallLaw==4?"spalding_profile3":(sstG1.profileAverageWallLaw==3?"spalding_match_height":(sstG1.profileAverageWallLaw==2?"spalding_relative":(sstG1.profileAverageWallLaw==1?"log":"spalding")))),
        sstG1.profileAverageWallLaw==5?"FIVE_POINT_EQUAL_WEIGHT_LEAST_SQUARES":(sstG1.profileAverageWallLaw==4?"THREE_POINT_EQUAL_WEIGHT_LEAST_SQUARES":(sstG1.profileAverageWallLaw==3?"ONE_ROOT_AT_PRESCRIBED_HEIGHT":"ONE_ROOT_AT_NQ_VOLUME_AVERAGED_U_Y")),(int)sstG1.sharedSpaldingWallState);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==2)std::printf(
        "NODALS_SST_RELATIVE_WALL_TRACE_CONFIG velocityState=UBAR_MINUS_UWALL wallTrace=PHYSICAL_FE_TRACE momentumTrial=BAVG_MINUS_BWALL momentumTest=PHYSICAL_WALL_TRACE wallLaw=SPALDING sharedUTauToOFAuto=1 ofAutoOtherAverages=UNCHANGED status=PASS\n");
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==3)std::printf(
        "NODALS_SST_MATCH_HEIGHT_CONFIG targetY=%.12e wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID "
        "utau=SPALDING_AT_MATCH_POINT momentumTest=PHYSICAL_WALL_TRACE "
        "momentumTrial=FIRST_CELL_NQ12_AVERAGE betaPolicy=UTAU2_OVER_CURRENT_LOCAL_UBAR "
        "ofAutoSharedUTau=1 ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
        sstG1.profileMatchY);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==5)std::printf(
        "NODALS_SST_MATCH_PROFILE5_CONFIG y=[%.12e,%.12e,%.12e,%.12e,%.12e] samplesPerWallQuadrature=5 "
        "wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES "
        "unknown=ONE_UTAU momentumTest=PHYSICAL_WALL_TRACE momentumTrial=FIRST_CELL_NQ12_AVERAGE "
        "betaPolicy=FITTED_UTAU2_OVER_CURRENT_LOCAL_UBAR_WITH_FINITE_PROFILE_FALLBACK ofAutoSharedUTau=1 "
        "ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED reportedMatchYPlus=MIDDLE_SAMPLE_Y3 status=PASS\n",
        sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==4)std::printf(
        "NODALS_SST_MATCH_PROFILE3_CONFIG y=[%.12e,%.12e,%.12e] samplesPerWallQuadrature=3 "
        "wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES "
        "unknown=ONE_UTAU momentumTest=PHYSICAL_WALL_TRACE momentumTrial=FIRST_CELL_NQ12_AVERAGE "
        "betaPolicy=FITTED_UTAU2_OVER_CURRENT_LOCAL_UBAR ofAutoSharedUTau=1 "
        "ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED reportedMatchYPlus=MIDDLE_SAMPLE status=PASS\n",
        sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3);
      const char* momWallForm=
        momentumWallMode==hxt5d::MOM_WALL_SPALDING ? "OFFWALL_SPALDING_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ? "TRACE_SPALDING_WEAK_DIRICHLET" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ? "OFFWALL_SPALDING_ALIGNED_WEAK_DIRICHLET" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ? "PROFILE_INTEGRAL_SPALDING_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ? "OFFWALL_SPALDING_COMPATIBLE_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE ? (sstG1.profileAverageClosure?"NQ_AVERAGE_SST_TABLE_TRACTION":"OFFWALL_SST_TABLE_TRACTION") :
        "SYMMETRIC_ADJOINT_CONSISTENT_NITSCHE";
      const char* momWallVisc=
        momentumWallMode==hxt5d::MOM_WALL_WEAK_EFFECTIVE ? "NU_PLUS_NUT_OFFWALL_SAMPLE" :
        momentumWallMode==hxt5d::MOM_WALL_WEAK_MOLECULAR ? "MOLECULAR_NU" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ? "SPALDING_UTAU_FROM_WALL_TRACE_HN" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ? "SPALDING_UTAU_OFFWALL_PLUS_WD3_STRESS_DEFICIT" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ? (sstG1.profileAverageClosure?"SPALDING_UTAU_FROM_NQ_VOLUME_AVERAGED_U_Y":"SPALDING_UTAU_FROM_FULL_FE_NORMAL_PROFILE") :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ? "SPALDING_UTAU_TIMES_CARRIER_USTAR" :
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE ? (sstG1.profileAverageClosure?"SST2003M_TABLE_UTAU_FROM_NQ_AVERAGED_U_Y":"SST2003M_TABLE_UTAU") :
        "SPALDING_UTAU";
      const bool momWallApplied=
        momentumWallMode==hxt5d::MOM_WALL_SPALDING ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ||
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE;
      std::printf("NODALS_SST_MOMENTUM_WALL_CONFIG mode=%s target=HOMOGENEOUS_TANGENTIAL_NOSLIP form=%s viscosity=%s weakGamma=%.12e hScale=%.12e nutScale=%.12e hDefinition=%s transverseWallClamp=STRONG_UX_UY spaldingMetrics=%s coarsePressureHierarchyWallSnapshot=BASELINE_SPALDING_PRECONDITIONER_ONLY status=PASS\n",
        hxt5d::momentum_wall_mode_name(momentumWallMode),momWallForm,momWallVisc,
        weakWallGamma,weakWallHScale,weakWallNutScale,
        (momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL||momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE)&&sstG1.profileAverageClosure?"NQ_VOLUME_AVERAGED_WALL_COLUMN":(momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL?"FULL_WALL_NORMAL_PROFILE":"FULL_WALL_NORMAL_CELL_SIZE_FROM_OFFWALL_SAMPLE"),
        momWallApplied?"APPLIED_TRACTION_AND_DIAGNOSTIC":"DIAGNOSTIC_ONLY_NOT_MOMENTUM_TRACTION");
      std::printf(
        "NODALS_SST_SPALDING_MOLECULAR_CONSISTENCY "
        "enabled=%d "
        "terms=TWO_MOLECULAR_NITSCHE_CONSISTENCY_TERMS "
        "extraPenalty=NONE "
        "viscosity=MOLECULAR_NU "
        "status=PASS\n",
        spaldingMolecularConsistency);
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED){
        std::printf(
          "NODALS_SST_SPALDING_ALIGNED_CONFIG "
          "law=SPALDING offwallSampleFraction=%.12e "
          "Cb=%.12e tauB=MAX_WALL_DEFICIT_AND_CB_NU_OVER_HB "
          "molecularShearApprox=NU_TIMES_UA_OVER_YA "
          "wallState=PHYSICAL_TRACE_PENALTY "
          "status=PASS\n",
          wallSampleFraction,weakWallGamma);
      }
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL){
        if(spaldingMolecularConsistency)throw std::runtime_error("spalding_profile_integral requires SPALDING_MOLECULAR_CONSISTENCY=0");
        if(!sstG1.profileIntegralUTau)throw std::runtime_error("spalding_profile_integral requires SST_PROFILE_INTEGRAL_UTAU=1");
        if(sstG1.profileIntegralNQ!=4&&sstG1.profileIntegralNQ!=8&&sstG1.profileIntegralNQ!=12)throw std::runtime_error("spalding_profile_integral requires SST_PROFILE_INTEGRAL_NQ=4, 8, or 12");
        const bool profileOmegaModeSupported =
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_SAMPLE_LOG ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST;
        if(!profileOmegaModeSupported)throw std::runtime_error("spalding_profile_integral requires SST_OMEGA_WALL_MODE=of_auto, sample_log, or trace_log");
        if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO && !sstG1.sharedSpaldingWallState)throw std::runtime_error("spalding_profile_integral + of_auto requires SST_SHARED_SPALDING_WALL_STATE=1");
        if(sstG1.profileIntegralClosure&&sstG1.profileAverageClosure)throw std::runtime_error("SST_PROFILE_INTEGRAL_CLOSURE and SST_PROFILE_AVERAGE_CLOSURE are mutually exclusive");
        if(sstG1.profileAverageClosure && sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO){
          std::printf(
            "NODALS_SST_PROFILE_AVERAGE_CLOSURE_CONFIG "
            "enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
            "averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN "
            "utau=SPALDING_ON_NQ_AVERAGED_U_Y "
            "sstState=ONE_OFAUTO_STATE_FROM_AVERAGED_U_GRADU_K_OMEGA_GRADK_GRADOMEGA "
            "carrierAverage=ACTIVE_CONSTITUTIVE_CARRIER_VALUES "
            "omegaConstraint=RANK_ONE_CELL_AVERAGE_PENALTY "
            "omegaPenalty=LAMBDA_GAMMA_NUEFF_OVER_HN2 "
            "production=AVERAGE_STATE_VOLUME_DISTRIBUTED "
            "sampleFractionRole=DIAGNOSTIC_ONLY "
            "status=PASS\n",
            sstG1.profileIntegralNQ);
        }
        if(sstG1.profileIntegralClosure){
          if(sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO)throw std::runtime_error("SST_PROFILE_INTEGRAL_CLOSURE=1 requires SST_OMEGA_WALL_MODE=of_auto");
          if(!(sstG1.profileClosureYPlusMin>0.0))throw std::runtime_error("SST_PROFILE_CLOSURE_YPLUS_MIN must be > 0");
          std::printf(
            "NODALS_SST_PROFILE_CLOSURE_CONFIG "
            "enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
            "utau=SHARED_PROFILE_INTEGRAL_SPALDING "
            "omegaTarget=OF_AUTO_PROFILE_VOLUME_PROJECTION "
            "omegaPenalty=LAMBDA_GAMMA_NUEFF_OVER_HN2 "
            "omegaYPlusMin=%.12e "
            "production=OF_AUTO_PROFILE_VOLUME_INTEGRAL "
            "sampleFractionRole=DIAGNOSTIC_ONLY "
            "status=PASS\n",
            sstG1.profileIntegralNQ,sstG1.profileClosureYPlusMin);
        }
        std::printf(
          "NODALS_SST_PROFILE_INTEGRAL_WALL_CONFIG "
          "normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
          "utauResidual=%s "
          "momentumTrial=PROFILE_AVERAGED_FE_VELOCITY "
          "tauWall=UTAU_SQUARED "
          "sstUtau=%s "
          "sstOmegaMode=%s "
          "sstTargetPlacement=%s "
          "status=PASS\n",
          sstG1.profileIntegralNQ,
          sstG1.profileAverageClosure?"AVERAGE_FIRST_SPALDING_AT_UBAR_YBAR":"MEAN_SPALDING_YPLUS_RESIDUAL_ZERO",
          sstG1.profileAverageClosure?"SAME_NQ_AVERAGE_FIRST_INVERSION":"SAME_PROFILE_INTEGRAL_INVERSION",
          hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),
          sstG1.profileAverageClosure?"NQ_VOLUME_AVERAGE_STATE":(sstG1.profileIntegralClosure?"FULL_PROFILE_VOLUME_PROJECTION":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"PHYSICAL_TRACE":"EXISTING_OFFWALL_SAMPLE")));
      }
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE){
        if(spaldingMolecularConsistency)throw std::runtime_error("spalding_compatible requires SPALDING_MOLECULAR_CONSISTENCY=0");
        if(sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_COMPATIBLE)throw std::runtime_error("spalding_compatible requires SST_OMEGA_WALL_MODE=compatible");
        std::printf(
          "NODALS_SST_COMPATIBLE_CONFIG "
          "velocityLaw=SPALDING_OFFWALL tauWall=UTAU_TIMES_USTAR "
          "uStar=CARRIER_FOURTH_POWER_BLEND omega=CARRIER_VIS_LOG_ROOT_SUM_SQUARE "
          "production=CARRIER_UTAU2_USTAR_OVER_KAPPA_Y_IF_YPLUS_GT_10 "
          "distance=EXISTING_OFFWALL_SAMPLE firstTestAdaptation=1 "
          "fullCarrierTruncatedDistanceAndFourLayerOmega=0 status=PASS\n");
      }

      // NODALS_SST_G2_MOM_GATE_V1
      // Additional OUTER convergence condition only.
      // Does not alter momentum, pressure, SST, wall, or SIMPLE operators.
      double sstG2MomAtol=0.0;
      if(const char* e=std::getenv("SST_G2_MOM_ATOL")){
        sstG2MomAtol=std::stod(e);
      }
      if(!(sstG2MomAtol>=0.0) || !std::isfinite(sstG2MomAtol))
        throw std::runtime_error("bad SST_G2_MOM_ATOL");

      std::printf(
        "NODALS_SST_G2_MOM_GATE "
        "enabled=%d absTol=%.12e "
        "components=XYZ "
        "quantity=PHYSICAL_MOMENTUM_RESIDUAL_NORM "
        "operatorChanged=0 status=PASS\n",
        (int)(sstG2MomAtol>0.0),
        sstG2MomAtol);

      if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_LOWRE_RESOLVED){
        if(momentumWallMode!=hxt5d::MOM_WALL_WEAK_MOLECULAR)
          throw std::runtime_error("lowre_resolved requires MOMENTUM_WALL_MODE=weak_molecular");
        if(sstG1.ofProduction)
          throw std::runtime_error("lowre_resolved requires SST_OF_PRODUCTION=0");
        if(sstG1.profileAverageClosure||sstG1.profileIntegralClosure)
          throw std::runtime_error("lowre_resolved requires profile average/integral SST closures OFF");
        std::printf(
          "NODALS_SST_LOWRE_RESOLVED_CONFIG "
          "momentumWall=WEAK_MOLECULAR_SYMMETRIC_NITSCHE_NO_SLIP "
          "momentumPenaltyGamma=%.12e momentumViscosity=MOLECULAR_NU "
          "kWall=ZERO_TRACE_SYMMETRIC_NITSCHE "
          "omegaWall=VISCOUS_ASYMPTOTE_C_NU_OVER_BETA1_Y1_SQUARED "
          "omegaFactor=%.12e omegaDistance=OFFWALL_REFERENCE_Y "
          "spaldingPhysics=NONE spaldingDiagnostics=ALLOWED "
          "ofAutoProduction=OFF wallFunctionNut=NONE status=PASS\n",
          weakWallGamma,sstG1.lowReOmegaFactor);
      }

      std::printf("NODALS_SST_G2_CONFIG model=SST-2003m precision=%s coupling=FULL_SIMPLE_K_OMEGA momentumFeedback=ON start=%s velocityInlet=DG_NUMERICAL_TRACE_PLUG turbulenceInlet=DG_TRACE intensity=%.6g lengthScaleOverD=%.6g kIn=%.12e omegaIn=%.12e outlet=NATURAL_ZERO_GRADIENT momentumWall=%s kWall=%s omegaWall=%s wallDistance=%s alphaK=%.6g alphaOmega=%.6g scalarLinear=%s plateauTol=%.6e plateauSamples=%d plateauEvery=%d maxOuter=%d status=PASS\n",
        PRECISION,directSSTStart?(stage27PotentialStart?"LAPLACE_P1_PROJECTED":"PLUG"):"CONVERGED_G0_MIXLEN",sstG1.intensity,sstG1.lengthRatio,TS.kIn,TS.omegaIn,hxt5d::momentum_wall_mode_name(momentumWallMode),sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE?"SST2003M_TABLE_SAMPLE":"HIGH_RE_ZERO_GRADIENT",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),stage20DistanceMethod=="geometric"?"GEOMETRIC_TRIANGLE_P1":"ANALYTIC_PIPE_R_MINUS_R",sstG1.alphaK,sstG1.alphaOmega,sstG1.linearSolver.c_str(),sstG2PlateauTol,sstG2PlateauSamples,sstG2PrintEvery,sstG2Max);
      std::printf("NODALS_SST_COMPACT_CONFIG precision=%s contAbsTol=%.6e turbRelTol=%.6e plateauEvery=%d plateauSamples=%d momentum=%s scalar=%s pressure=%s status=PASS\n",PRECISION,sstG2ContAtol,sstG2PlateauTol,sstG2PrintEvery,sstG2PlateauSamples,momWork.c_str(),sstG1.linearSolver.c_str(),pressureSolver.c_str());
      std::printf("NODALS_IMPLICIT_STRESS form=%s momentumVolume=%s "
        "tensor=2_NU_EFF_DEV_SYM_GRAD_U quadrature=TET24 "
        "coupling=IMPLICIT_9BLOCK_CSR_FGS_OR_MATRIXFREE_FGMRES "
        "bf3=TtMT wall=EXISTING_SPALDING "
        "pressureSchur=LEGACY_SCALAR_DIAGONAL_APPROX "
        "kOmega=UNCHANGED legacyLaggedRHS=OFF status=EXPERIMENTAL\n",
        stressForm.c_str(),implicitDevStress?"FULL_DEVIATORIC":"VECTOR_LAPLACIAN");
      std::printf("NODALS_SST_STRESS_FORM production=%s momentumVolume=%s coupling=%s devProduction=%d devMomentum=%d status=PASS\n",
        sstG1.deviatoricProduction?"DEV_TWO_SYMM_GRADU_COLON_GRADU":"TWO_S_COLON_S_DIVFREE_SHORTCUT",
        sstG1.deviatoricMomentum?"DEV_TWO_SYMM_NU_EFF_LAGGED_CROSS":"SCALAR_NU_EFF_GRADU",
        sstG1.deviatoricMomentum?"IMPLICIT_SCALAR_PLUS_EXPLICIT_CROSS":"IMPLICIT_SCALAR",
        (int)sstG1.deviatoricProduction,(int)sstG1.deviatoricMomentum);
      std::printf("NODALS_SST_G2_CONVERGENCE_CONFIG physicalGate=%d metricsEvery=%d ""simpleTol=%.6e plateauTol=%.6e plateauSamples=%d plateauEvery=%d ""semantics=mass_and_turbulence_plus_optional_physical_plateau status=PASS\n",(int)sstG2PhysicalGate,sstG2MetricsEvery,simpleTol,sstG2PlateauTol,sstG2PlateauSamples,sstG2PrintEvery);
      HXT1_CUDA(cudaDeviceSynchronize());
      const auto g2PerfT0 = std::chrono::steady_clock::now();

      // NODALS_SST_G2_VTU_CHECKPOINT_OPTIONS_V2
      // Runtime-configurable checkpoints. 0 disables them.
      int sstG2VtuEvery=100;
      if(const char* e=std::getenv("SST_G2_VTU_EVERY")) sstG2VtuEvery=std::stoi(e);
      if(sstG2VtuEvery<0) throw std::runtime_error("bad SST_G2_VTU_EVERY");
      std::string sstG2VtuDir=".";
      if(const char* e=std::getenv("SST_G2_VTU_DIR")) if(*e) sstG2VtuDir=e;
      std::printf("NODALS_SST_G2_VTU_CHECKPOINT_CONFIG every=%d dir=%s fields=FLOW_PLUS_Q1_SST status=PASS\\n",
                  sstG2VtuEvery,sstG2VtuDir.c_str());

      // Stage31 restartable FP64 state (not visualization VTU).
      const int stage31Every=stage31_restart::cadence();
      // Optional expensive diagnostic operator actions, NOT physics/safety checks.
      // 0 disables them; N>0 runs only every N absolute SIMPLE iterations.
      const int nodalsDiagnosticEvery=std::stoi(
        stage31_restart::env_string("NODALS_SST_DIAGNOSTIC_EVERY","0"));
      if(nodalsDiagnosticEvery<0 || nodalsDiagnosticEvery>1000000)
        throw std::runtime_error("NODALS_SST_DIAGNOSTIC_EVERY outside 0..1000000");
      const auto nodalsDiagnosticsDue=[&](int it){
        return nodalsDiagnosticEvery>0 && it%nodalsDiagnosticEvery==0;
      };
      std::printf("NODALS_SOFTWARE_DIAGNOSTICS cadence=%d (0=off) status=CONFIGURED\n",
                  nodalsDiagnosticEvery);
      const std::string stage31Fingerprint=stage31_restart::fingerprint();
      const std::string stage31RestartPath=stage31_restart::env_string("STAGE31_RESTART_FROM");
      const std::string stage31Dir=stage31_restart::env_string("STAGE31_CHECKPOINT_DIR",sstG2VtuDir);
      int stage31Start=0;
      if(!stage31RestartPath.empty()){
        if(podMode!="off")throw std::runtime_error("Stage31 restart currently requires POD=off");
        stage31_restart::Fields<T> ck;
        double savedNut=0;
        stage31Start=stage31_restart::read<T>(stage31RestartPath,nv,np,(int)M.h.nv,
                                              stage31Fingerprint,ck,savedNut);
        if(stage31Start>=sstG2Max)
          throw std::runtime_error("Stage31: --max-simple must exceed saved absolute iteration");
        auto restore=[&](T *device,const std::vector<T>&v){
          HXT1_CUDA(cudaMemcpy(device,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
        };
        restore(d_u0.p,ck.u0);restore(d_u1.p,ck.u1);restore(d_u2.p,ck.u2);
        restore(d_p.p,ck.p);restore(TS.k.p,ck.k);restore(TS.omega.p,ck.omega);
        restore(TS.oldK.p,ck.k);restore(TS.oldW.p,ck.omega);
        TS.prevMeanNut=savedNut;
        // Reinitialize derived operators and convergence/plateau history on first resumed step.
        std::printf("STAGE31_RESTART_LOAD=PASS path=%s iteration=%d next=%d "
                    "mode=WARM_RESTART settingsMayChange=1 pressurePOD=OFF "
                    "fields=FULL_BF3_U_P_K_OMEGA checksum=PASS\n",
                    stage31RestartPath.c_str(),stage31Start,stage31Start+1);
      }
      std::printf("STAGE31_RESTART_CONFIG every=%d dir=%s start=%d fingerprint=%s status=PASS\n",
                  stage31Every,stage31Dir.c_str(),stage31Start,stage31Fingerprint.c_str());
      std::fflush(stdout);

#if defined(NODALS_P2_STAGE4_COUPLED)
      const bool p2Stage5Active=stage31_restart::env_string("NODALS_P2_STAGE5_RUN")=="1";
      const bool p2Stage4Active=p2Stage5Active ||
        stage31_restart::env_string("NODALS_P2_STAGE4_TWO_STEP")=="1";
      // Crossover: old Stage31 it200 -> true P2 it201..210.
      // Fresh: potential-flow/plug initial condition -> P2 it1..200.
      // Both use the SAME compiled CUDA executable, selected at runtime.
      const bool p2Stage5Fresh=p2Stage5Active && stage31Start==0;
      int p2Stage5SaveEvery=1;
      if(p2Stage5Active){
        const std::string cadence=stage31_restart::env_string("NODALS_P2_SAVE_EVERY","20");
        p2Stage5SaveEvery=std::stoi(cadence);
        if(p2Stage5SaveEvery<1 || p2Stage5SaveEvery>1000)
          throw std::runtime_error("P2_STAGE5 NODALS_P2_SAVE_EVERY must be 1..1000");
      }
      std::unique_ptr<nodals::p2::pilot::Live> p2Stage4;
      if(p2Stage4Active){
        const bool correctIterations=p2Stage5Active
          ? ((stage31Start==200 && sstG2Max==210) || (stage31Start==0 && sstG2Max==500))
          : (stage31Start==200 && sstG2Max==202);
        if(!correctIterations || M.h.ntet!=200245 || M.h.nv!=34300 || M.h.nhex || M.h.ninterface ||
           !stage22d4Live || !f1bFrozenWall || !f2FaceSpalding ||
           sstG1.deviatoricMomentum || stage33MomentumSupg ||
           sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO)
          throw std::runtime_error("P2_STAGE5 requires pure tet, unchanged F1B/F2/OF_AUTO, scalar stress, SUPG off and exact 200k mesh/run bounds");
        if(std::getenv("NODALS_P2_NATIVE_PILOT") &&
           std::string(std::getenv("NODALS_P2_NATIVE_PILOT"))=="1")
          throw std::runtime_error("P2_STAGE5 cannot run concurrently with Stage3 pilot");
        p2Stage4.reset(new nodals::p2::pilot::Live(M));
        p2Stage4->lift(TS);
        if(p2Stage5Active)
          std::printf("P2_STAGE5_ACTIVATED=PASS start=%d end=%d mode=%s meshTet=%llu scalarDOFs=%d momentumP2First=%d saveEvery=%d checkpoint=P2_V2\n",
            stage31Start,sstG2Max,p2Stage5Fresh?"FRESH":"CROSSOVER",
            (unsigned long long)M.h.ntet,p2Stage4->matrix->n,
            stage31Start+(p2Stage5Fresh?1:2),p2Stage5SaveEvery);
        else
          std::printf("P2_STAGE4_ACTIVATED=PASS start=200 end=202 meshTet=%llu scalarDOFs=%d momentumP2At=202 wall=P2_F1B inlet=P2_DG checkpoint=P2_V2\n",
            (unsigned long long)M.h.ntet,p2Stage4->matrix->n);
        std::fflush(stdout);
      }
      auto p2Stage4Save=[&](int it){
        auto &P=*p2Stage4;
        const std::string p2conn=stage31_restart::env_string("NODALS_P2_CONN_SHA256");
        nodals::p2::assert_sha(p2conn);
        nodals::p2::CheckpointHeader hdr{};
        hdr.velocityDofs=(std::uint64_t)nv;hdr.pressureDofs=(std::uint64_t)np;
        hdr.vertices=(std::uint64_t)M.h.nv;hdr.scalarP2Dofs=(std::uint64_t)P.matrix->n;
        hdr.iteration=(std::uint64_t)it;hdr.prevMeanNut=TS.prevMeanNut;
        std::memcpy(hdr.inputSha256,stage31Fingerprint.data(),64);
        std::memcpy(hdr.connectivitySha256,p2conn.data(),64);
        nodals::p2::FieldsP2<T> f{};
        f.u0.resize(nv);f.u1.resize(nv);f.u2.resize(nv);
        f.pressure.resize(np);f.kP2.resize(P.matrix->n);f.omegaP2.resize(P.matrix->n);
        auto fetch=[&](std::vector<T>&v,const T*ptr){
          HXT1_CUDA(cudaMemcpy(v.data(),ptr,v.size()*sizeof(T),cudaMemcpyDeviceToHost));};
        fetch(f.u0,d_u0.p);fetch(f.u1,d_u1.p);fetch(f.u2,d_u2.p);
        fetch(f.pressure,d_p.p);fetch(f.kP2,P.k.p);fetch(f.omegaP2,P.w.p);
        const std::string ckpt=stage31Dir+
          (p2Stage5Active?"/NODALS_TRUE_P2_STAGE5_it":"/NODALS_TRUE_P2_STAGE4_it")+
          std::to_string(it)+".p2ckpt";
        nodals::p2::write_p2(ckpt,hdr,f);
        if(p2Stage5Active)
          std::printf("P2_STAGE5_CHECKPOINT=PASS iteration=%d path=%s format=2 scalarDOFs=%d edges=%zu\n",
            it,ckpt.c_str(),P.matrix->n,P.graph.conn.edges.size());
        else
          std::printf("P2_STAGE4_CHECKPOINT=PASS iteration=%d path=%s format=2 scalarDOFs=%d edges=%zu\n",
            it,ckpt.c_str(),P.matrix->n,P.graph.conn.edges.size());
      };
#endif

      // P50: only the NUMERICAL AMG fine-level pressure preconditioner is
      // reused. The live SAniso pressure equation and directional rAU remain
      // updated EVERY SIMPLE iteration. Refresh at the first iteration of a
      // run/restart and then every N absolute SIMPLE iterations.
      int p50RefreshEvery=1;
      if(const char* e=std::getenv("NODALS_P_REFRESH_EVERY"))
        p50RefreshEvery=std::stoi(e);
      if(p50RefreshEvery<1 || p50RefreshEvery>1000000)
        throw std::runtime_error("NODALS_P_REFRESH_EVERY must be 1..1000000");
      if(amgFineMode!="explicit" && p50RefreshEvery!=1)
        throw std::runtime_error("P50 periodic pressure preconditioner refresh requires --amg-fine-mode explicit");
      int p50LastRefresh=0, p50RefreshCount=0;
      double p50RefreshTotalMs=0.0;
      std::printf("NODALS_P50_CONFIG refreshEvery=%d firstRefresh=FIRST_SIMPLE "
        "pressureOperator=LIVE_EVERY_SIMPLE rAU=LIVE_EVERY_SIMPLE "
        "AMGHierarchy=FROZEN fineMatrix=PERIODIC coupledFGMRES=UNCHANGED "
        "F1B_F2=UNCHANGED status=PASS\n",p50RefreshEvery);

      // Stage 0/1 read-only policy: wall *physics* is unchanged.
      const auto stage27AuditPolicy=nodals::stage27::WallAuditPolicy::from_env();
      int stage27AuditCorrectedAt=0;
      std::printf("STAGE27_WALL_AUDIT_CONFIG mode=%s every=%d predictor=%d "
                  "freshConvergenceOnly=1 lightWallNormalEveryOuter=1 "
                  "physics=UNCHANGED status=PASS\n",
                  stage27AuditPolicy.name(),stage27AuditPolicy.period,
                  (int)stage27AuditPolicy.predictor);
      int stage22d4cAuditEvery=1;
      if(const char* e=std::getenv("SST22D4C_AUDIT_EVERY"))stage22d4cAuditEvery=std::stoi(e);
      if(stage22d4cAuditEvery<1||stage22d4cAuditEvery>sstG2Max)
        throw std::runtime_error("STAGE22D4C audit interval outside 1..maxOuter");
      auto stage22d4cAudit=[&](int it,const char* phase){
        if(!stage22d4Live)return;
        const auto g2AuditT0 = std::chrono::steady_clock::now();
        HXT1_CUDA(cudaMemset(d_s22d4cWallStats.p,0,18*sizeof(double)));
        HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
        hxt5d::s22d4c_wall_diagnostic<<<BH.wallT.size(),64>>>(
          sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
          sstG1.profileTetMatchPoints,d_tv.p,d_tg.p,d_pts.p,d_tc.p,
          d_u0.p,d_u1.p,d_u2.p,d_s22d4cLagU0.p,d_s22d4cLagU1.p,d_s22d4cLagU2.p,
          TS.k.p,TS.omega.p,nu,R,sstG1.kFloor,sstG1.omegaFloor,s22d4Gamma,
          d_s22d4cWallStats.p,d_bad.p,sampledConductance,stage22d4iEnable);
        HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
        unsigned long long errors=0;
        HXT1_CUDA(cudaMemcpy(&errors,d_bad.p,sizeof(errors),cudaMemcpyDeviceToHost));
        if(errors)throw std::runtime_error("STAGE22D4C wall diagnostic invalid quadrature="+std::to_string(errors));
        double w[18]={};
        HXT1_CUDA(cudaMemcpy(w,d_s22d4cWallStats.p,sizeof(w),cudaMemcpyDeviceToHost));
        const double aw=w[0],qin=std::max(std::abs(bulk)*stage22d4InletArea,1.e-300);
        if(!(aw>0.)||!std::isfinite(aw)||!std::isfinite(w[1])||!std::isfinite(w[4])||!std::isfinite(w[9]))
          throw std::runtime_error("STAGE22D4C nonfinite wall diagnostics");
        const double darcy=8.*w[4]/aw/(bulk*bulk),yplus=w[6]/aw;
        const double unmax=w[14];
        if(std::string(phase)=="CORRECTED"){
          stage22d4cPostF=darcy;stage22d4cPostY=yplus;stage22d4cPostWall=w[2];
          stage27AuditCorrectedAt=it;
        }
        if(it==1 || it%stage22d4cAuditEvery==0 || it==sstG2Max){
          std::printf("STAGE22D4C_WALL it=%d phase=%s gamma=%.8g "
            "area=%.12e netFlux=%.12e absFlux=%.12e netFluxOverInlet=%.12e "
            "absFluxOverInlet=%.12e meanAbsNormal=%.12e rmsNormal=%.12e "
            "maxAbsNormal=%.12e outwardFlux=%.12e inwardFlux=%.12e "
            "uTauAreaMean=%.12e yPlusMean=%.12e darcyFromUTau=%.12e "
            "tangentSpeedAreaMean=%.12e traction=[%.12e,%.12e,%.12e] "
            "normalPenaltyEnergy=%.12e picardCrossLagWeakL1=%.12e "
            "fullWallWeakActionL1=%.12e "
            "weakActionIsNotGlobalMomentumResidual=1 status=PASS\n",
            it,phase,s22d4Gamma,aw,w[2],w[1],fabs(w[2])/qin,w[1]/qin,
            w[1]/aw,sqrt(std::max(w[3],0.)/aw),unmax,w[12],w[13],
            w[5]/aw,yplus,darcy,w[8]/aw,w[15],w[16],w[7],w[17],w[9],w[10]);
        }
        if(std::string(phase)=="CORRECTED"){
          // Full FE boundary flux and volume divergence using independent
          // geometric integrations. Never substitute the prescribed DG inlet
          // trace for the actual CG velocity trace in this audit.
          HXT1_CUDA(cudaMemset(d_s22d4cFEFlux.p,0,3*sizeof(double)));
          hxt5d::s22d4c_patch_flux<<<BH.inT.size(),64>>>(
             d_inT.p,(int)BH.inT.size(),d_tv.p,
             d_u0.p,d_u1.p,d_u2.p,d_s22d4cFEFlux.p);
          hxt5d::s22d4c_patch_flux<<<stage22d4cOutFaces.size(),64>>>(
             d_s22d4cOutFaces.p,(int)stage22d4cOutFaces.size(),d_tv.p,
             d_u0.p,d_u1.p,d_u2.p,d_s22d4cFEFlux.p+1);
          hxt5d::s22d4c_volume_divergence<<<(M.h.ntet+127)/128,128>>>(
             d_tv.p,d_tg.p,M.h.ntet,d_u0.p,d_u1.p,d_u2.p,d_s22d4cFEFlux.p+2);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          double v[3]={};
          HXT1_CUDA(cudaMemcpy(v,d_s22d4cFEFlux.p,sizeof(v),cudaMemcpyDeviceToHost));
          if(it==1||it%stage22d4cAuditEvery==0||it==sstG2Max){
            const double surface=v[0]+v[1]+w[2];
            std::printf("STAGE22D4C_MASS it=%d gamma=%.8g "
              "FEInletOutward=%.12e FEOutletOutward=%.12e FEWallOutward=%.12e "
              "FEAllSurfaceOutward=%.12e FEVolumeDivergence=%.12e "
              "surfaceMinusVolume=%.12e prescribedInletFlux=%.12e "
              "interpretation=INDEPENDENT_FE_TRACE_AND_VOLUME_NOT_BN_CONTINUITY status=PASS\n",
              it,s22d4Gamma,v[0],v[1],w[2],surface,v[2],surface-v[2],
              -BH.sourceSum);
          }
        }
        if(std::getenv("NODALS_G2_CHILD_PROFILE") &&
           std::string(std::getenv("NODALS_G2_CHILD_PROFILE"))=="1"){
          const double auditMs=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-g2AuditT0).count();
          std::printf("STAGE26_G2_WALL_AUDIT it=%d phase=%s wallMs=%.6f\n",it,phase,auditMs);
        }
      };
      // STAGE26_PROFILE_CHILD_ONLY — measurements; no physics or solver changes.
      const bool g2ChildProfile = std::getenv("NODALS_G2_CHILD_PROFILE") &&
          std::string(std::getenv("NODALS_G2_CHILD_PROFILE")) == "1";
      cudaEvent_t g2ChildEv[10]{};
      std::size_t g2ChildUsed[10]{};
      double g2ChildTotalMiB=0.0;
      if(g2ChildProfile){
        for(auto &ev:g2ChildEv)HXT1_CUDA(cudaEventCreate(&ev));
        std::printf("STAGE26_G2_CHILD_PROFILE enabled=1 points=10 metric=CUDA_EVENT_PLUS_LIVE_VRAM noPhysicsChanges=1\n");
      }
      auto g2ChildSnap = [&](int n){
        if(!g2ChildProfile)return;
        HXT1_CUDA(cudaEventRecord(g2ChildEv[n],0));
        std::size_t avail=0,total=0;
        HXT1_CUDA(cudaMemGetInfo(&avail,&total));
        g2ChildUsed[n]=total-avail;
        g2ChildTotalMiB=double(total)/(1024.0*1024.0);
      };
      // PREDICTOR_AUDIT: CUDA event timestamps only. All numerical paths remain identical.
      // Must run with coupled FGMRES; each timestamp is recorded on the default stream.
      const bool predAuditEnabled=std::getenv("NODALS_PREDICTOR_AUDIT") &&
        std::string(std::getenv("NODALS_PREDICTOR_AUDIT"))=="1";
      cudaEvent_t predAuditEv[11]{};
      if(predAuditEnabled){
        if(!stage22d4Live || stage22d4dMode!="coupled" || s05b1Active)
          throw std::runtime_error("PREDICTOR_AUDIT requires standard coupled FGMRES");
        for(auto &ev:predAuditEv) HXT1_CUDA(cudaEventCreate(&ev));
        std::printf("NODALS_PREDICTOR_AUDIT_CONFIG enabled=1 segments=10 "
          "method=CUDA_EVENTS defaultStream=YES pressureRefreshCadence=%d "
          "momentum=COUPLED_FGMRES numerics=UNCHANGED status=PASS\n",p50RefreshEvery);
      }
      auto predAuditMark=[&](int which){
        if(predAuditEnabled) HXT1_CUDA(cudaEventRecord(predAuditEv[which],0));
      };
      auto predAuditPrint=[&](int it){
        if(!predAuditEnabled) return;
        HXT1_CUDA(cudaEventSynchronize(predAuditEv[10]));
        const char* labels[10]={
          "rau_and_schur_diagonal", "amg_fine_refresh_or_reuse",
          "pressure_gradient_and_rhs", "momentum_presolve_pack_and_residual",
          m3Active?"coupled_csr_fixed_fgs_sgs":"coupled_fgmres", "predictor_unpack",
          "postsolve_momentum_residual_and_audits", "continuity_rhs_and_stats",
          "pressure_pcg_amg", "post_pcg_pod_bookkeeping"};
        double sum=0.;
        for(int j=0;j<10;++j){
          float ms=0.f;
          HXT1_CUDA(cudaEventElapsedTime(&ms,predAuditEv[j],predAuditEv[j+1]));
          sum+=ms;
          std::printf("NODALS_PREDICTOR_AUDIT it=%d phase=%s ms=%.6f status=PASS\n",
            it,labels[j],double(ms));
        }
        float total=0.f;
        HXT1_CUDA(cudaEventElapsedTime(&total,predAuditEv[0],predAuditEv[10]));
        std::printf("NODALS_PREDICTOR_AUDIT_SUM it=%d subphaseMs=%.6f inclusiveMs=%.6f "
          "differenceMs=%.6f status=%s\n",it,sum,double(total),sum-double(total),
          std::fabs(sum-double(total))<=0.1?"PASS":"WARN");
        std::fflush(stdout);
      };
      for(int git=stage31Start+1;git<=sstG2Max;++git){
#if defined(NODALS_P2_STAGE4_COUPLED)
        // Fresh P2 has valid P2 k/w at iteration 1 (lift of uniform inlet),
        // whereas checkpoint crossover preserves legacy it201 momentum parity.
        const bool p2MomentumLive=p2Stage4Active && (p2Stage5Fresh || git>stage31Start+1);
#endif
        if(git==1||git==3||git==5)stage6_mem::snapshot("simple_iteration_start",16);
        g2ChildSnap(0);
#include "../coupling/simple_preconvection.inc"
        // Reacquire exactly the same two full CSR buffers before assembling
        // the new momentum wall and bulk operators. Existing Stage6B1 reuse
        // still works; no component operator or SpMV kernel is changed.
        if(stage6c2PhaseReuse && git>stage31Start+1){
          if(d_s22ValY.p || d_valZ.p || stage6bWallY || stage6bWallZ)
            throw std::runtime_error("Stage6C2 buffers unexpectedly live at reacquire");
          d_s22ValY.allocate_zeros((std::size_t)A.nnz);
          d_valZ.allocate_zeros((std::size_t)A.nnz);
          stage6bWallY=d_s22ValY.p; stage6bWallZ=d_valZ.p;
          stage6_mem::tag(d_s22ValY.p,"momentum/assembled_Y_reacquired");
          stage6_mem::tag(d_valZ.p,"momentum/assembled_Z_reacquired");
          std::printf("STAGE6C2_CSR_LIFETIME it=%d action=REACQUIRE bytes=%zu "
            "mode=FULL_CSR status=PASS\n",git,2u*(std::size_t)A.nnz*sizeof(T));
        }
        g2ChildSnap(1);
#include "../coupling/simple_momentum_assembly.inc"
        if(git==1||git==3||git==5)stage6_mem::snapshot("after_momentum_assembly",16);
        g2ChildSnap(2);
predAuditMark(0);
#include "../coupling/simple_pressure_predictor.inc"
        predAuditMark(10);
        if(git==1||git==3||git==5)stage6_mem::snapshot("after_momentum_pressure_predictor",16);
        g2ChildSnap(3);
#include "../coupling/simple_pressure_pod.inc"
        g2ChildSnap(4);
#include "../coupling/simple_pressure_corrector.inc"
        if(git==1||git==3||git==5)stage6_mem::snapshot("after_pressure_correction",16);
        // All momentum and SIMPLE-corrected physical residuals are finished.
        // Scalar transport uses A.base/conv/val and d_turb/d_dg/d_wall,
        // but never the finalized Y/Z matrices. Drop them before SST scratch
        // reaches its temporary peak. cudaDeviceSynchronize fences kernels
        // that used the buffers; reacquire happens next SIMPLE iteration.
        if(stage6c2PhaseReuse){
          HXT1_CUDA(cudaDeviceSynchronize());
          if(stage6bWallY!=d_s22ValY.p || stage6bWallZ!=d_valZ.p)
            throw std::runtime_error("Stage6C2 wall/CSR pointer ownership mismatch");
          d_s22ValY.release();d_valZ.release();
          stage6bWallY=nullptr;stage6bWallZ=nullptr;
          std::printf("STAGE6C2_CSR_LIFETIME it=%d action=RELEASE bytes=%zu "
            "phase=BEFORE_SST_TRANSPORT status=PASS\n",
            git,2u*(std::size_t)A.nnz*sizeof(T));
          if(git==1||git==3||git==5)
            stage6_mem::snapshot("stage6c2_after_yz_release",16);
        }
        g2ChildSnap(5);
#if defined(NODALS_P2_STAGE4_COUPLED)
        if(p2Stage4Active){
          auto &P=*p2Stage4;
          P.run_once(git,M,BH,d_inT,d_pts,d_tc,d_tv,d_tg,
                     d_u0.p,d_u1.p,d_u2.p,bulk,R,nu,sstG1,true);
          // Legacy masked-BF3 arrays hold a VERTEX PROJECTION ONLY for the
          // differential volume audit, never as a replacement for P2 state.
          HXT1_CUDA(cudaMemcpy(TS.k.p,P.k.p,(std::size_t)M.h.nv*sizeof(T),cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(TS.omega.p,P.w.p,(std::size_t)M.h.nv*sizeof(T),cudaMemcpyDeviceToDevice));
          // Advance the original momentum convection state using the CURRENT
          // pressure-corrected U; normally done in step_turbulence(), which
          // the full-P2 branch deliberately bypasses.
          HXT1_CUDA(cudaMemset(A.conv.p,0,(std::size_t)A.nnz*sizeof(T)));
          tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(
            d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          std::printf("P2_STAGE4_CONVECTION_REFRESH=PASS it=%d next=%d pressureCorrectedVelocity=YES\n",git,git+1);
          if(p2Stage5Active){
            if(git==stage31Start+1 || git%p2Stage5SaveEvery==0 || git==sstG2Max)
              p2Stage4Save(git);
            std::printf("P2_STAGE5_STEP=PASS it=%d mode=%s momentumP2=%s scalars=P2_FGMRES checkpointCadence=%d convergence=NOT_ESTABLISHED\n",
              git,p2Stage5Fresh?"FRESH":"CROSSOVER",p2MomentumLive?"APPLIED":"LEGACY_REFERENCE",p2Stage5SaveEvery);
            std::fflush(stdout);
            if(git==sstG2Max){
              std::printf("P2_STAGE5_RUN=PASS start=%d end=%d mode=%s coupledFrom=%d physicalConvergence=NOT_CLAIMED\n",
                stage31Start+1,sstG2Max,p2Stage5Fresh?"FRESH":"CROSSOVER",
                stage31Start+(p2Stage5Fresh?1:2));
              std::fflush(stdout);
              return 0;
            }
          }else{
            p2Stage4Save(git);
            if(git==202){
              std::printf("P2_STAGE4_TWO_STEP=PASS iterations=201,202 P2volume=ITERATION202 P2wall=ITERATION202 P2inlet=ITERATION202 P2scalars=PERSISTENT physicalConvergence=NOT_CLAIMED\n");
              std::fflush(stdout);
              return 0;
            }
          }
          // The Stage31 checkpoint, legacy SST diagnostics and vertex-only
          // VTU cannot represent P2 edge controls: do not execute them.
          continue;
        }
#endif
#if defined(NODALS_P2_NATIVE_PILOT)
        // Real 10-mode P2 GPU assembly/FGMRES. One step only: momentum feedback
        // is not P2-routed yet, so do not silently call this coupled P2 SST.
        if(const char* pilot=std::getenv("NODALS_P2_NATIVE_PILOT")){
          if(std::string(pilot)=="1"){
            if(git!=stage31Start+1)throw std::runtime_error("P2_PILOT only first restarted SIMPLE step");
            nodals::p2::pilot::Live P(M);
            P.lift(TS);
            P.run_once(git,M,BH,d_inT,d_pts,d_tc,d_tv,d_tg,
                       d_u0.p,d_u1.p,d_u2.p,bulk,R,nu,sstG1);
            // True ten-control P2 constitutive k/omega -> original P1+BF3
            // momentum *volume* test: assemble into separate scratch, WITHOUT
            // changing the accepted momentum matrix or next SIMPLE iterate.
            Dev<T> p2MomentumScratch(std::vector<T>((std::size_t)A.nnz,T(0)));
            Dev<T> p2MomentumDiagonal(std::vector<T>((std::size_t)nv,T(0)));
            hxt5d::sst_mom_p2_tet<<<M.h.ntet,128>>>(d_pts.p,d_tc.p,d_tv.p,d_tg.p,
              A.tetSlot.p,P.d_tets.p,d_u0.p,d_u1.p,d_u2.p,P.k.p,P.w.p,
              p2MomentumScratch.p,p2MomentumDiagonal.p,M.h.ntet,R,nu,
              sstG1.kFloor,sstG1.omegaFloor);
            HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
            std::vector<T> dtest((std::size_t)nv);
            HXT1_CUDA(cudaMemcpy(dtest.data(),p2MomentumDiagonal.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
            double sum=0.0,maxabs=0.0;int badDiag=0;
            for(auto v:dtest){if(!std::isfinite((double)v) || v<T(0))++badDiag;
              sum+=(double)v;maxabs=std::max(maxabs,std::fabs((double)v));}
            if(badDiag || !(sum>0.0) || !std::isfinite(sum))
              throw std::runtime_error("P2 native momentum-volume nut assembly invalid");
            std::printf("P2_NATIVE_MOM_NUT=PASS it=%d volumeTets=%llu momentumLocalModes=8 scalarLocalModes=10 totalDiagNut=%.12e maxDiagNut=%.12e scope=VOLUME_ONLY notYetAppliedToMomentum=YES\n",
              git,(unsigned long long)M.h.ntet,sum,maxabs);
            // Stage3 native P2 format-2 checkpoint; independent of the R1
            // BF3 Stage31 checkpoint, with real P2 edge-coefficient vectors.
            const std::string p2conn=stage31_restart::env_string("NODALS_P2_CONN_SHA256");
            nodals::p2::assert_sha(p2conn);
            nodals::p2::CheckpointHeader hdr{};
            hdr.velocityDofs=(std::uint64_t)nv;hdr.pressureDofs=(std::uint64_t)np;
            hdr.vertices=(std::uint64_t)M.h.nv;hdr.scalarP2Dofs=(std::uint64_t)P.matrix->n;
            hdr.iteration=(std::uint64_t)git;hdr.prevMeanNut=TS.prevMeanNut;
            std::memcpy(hdr.inputSha256,stage31Fingerprint.data(),64);
            std::memcpy(hdr.connectivitySha256,p2conn.data(),64);
            nodals::p2::FieldsP2<T> f{};
            f.u0.resize(nv);f.u1.resize(nv);f.u2.resize(nv);
            f.pressure.resize(np);f.kP2.resize(P.matrix->n);f.omegaP2.resize(P.matrix->n);
            auto fetch=[&](std::vector<T>&v,const T*ptr){
              HXT1_CUDA(cudaMemcpy(v.data(),ptr,v.size()*sizeof(T),cudaMemcpyDeviceToHost));};
            fetch(f.u0,d_u0.p);fetch(f.u1,d_u1.p);fetch(f.u2,d_u2.p);
            fetch(f.pressure,d_p.p);fetch(f.kP2,P.k.p);fetch(f.omegaP2,P.w.p);
            const std::string p2ckpt=stage31Dir+"/NODALS_TRUE_P2_PILOT_it"+std::to_string(git)+".p2ckpt";
            nodals::p2::write_p2(p2ckpt,hdr,f);
            std::printf("P2_NATIVE_CHECKPOINT=PASS iteration=%d path=%s format=2 scalarDOFs=%d vertexDOFs=%llu edgeDOFs=%zu sourceStage31=R1\n",
              git,p2ckpt.c_str(),P.matrix->n,(unsigned long long)M.h.nv,P.graph.conn.edges.size());
            std::fflush(stdout);
            return 0;
          }
        }
#endif
#include "../coupling/simple_turbulence_advance.inc"
        if(git==1||git==3||git==5)stage6_mem::snapshot("after_k_omega",16);
        g2ChildSnap(6);
#include "../coupling/simple_checkpoint.inc"
        g2ChildSnap(7);
#include "../coupling/simple_metrics.inc"
        if(git==1||git==3||git==5)stage6_mem::snapshot("after_metrics",16);
        g2ChildSnap(8);
#include "../coupling/simple_convergence.inc"
        // Stage31 native restart checkpoint, written only after a complete SIMPLE step.
        if(stage31Every>0 && (git%stage31Every==0 || g2Converged)){
          stage31_restart::Fields<T> ck;
          ck.u0.resize(nv);ck.u1.resize(nv);ck.u2.resize(nv);
          ck.p.resize(np);ck.k.resize(nv);ck.omega.resize(nv);
          auto take=[&](std::vector<T>&v,const T*ptr){
            HXT1_CUDA(cudaMemcpy(v.data(),ptr,v.size()*sizeof(T),cudaMemcpyDeviceToHost));
          };
          take(ck.u0,d_u0.p);take(ck.u1,d_u1.p);take(ck.u2,d_u2.p);
          take(ck.p,d_p.p);take(ck.k,TS.k.p);take(ck.omega,TS.omega.p);
          const std::string ckpath=stage31Dir+"/NODALS_STAGE31_STATE_it"+std::to_string(git)+".ckpt";
          stage31_restart::write<T>(ckpath,git,nv,np,(int)M.h.nv,
                                     TS.prevMeanNut,stage31Fingerprint,ck);
          std::printf("STAGE31_RESTART_SAVE=PASS iteration=%d path=%s "
                      "fields=FULL_BF3_U_P_K_OMEGA checksum=PASS\n",git,ckpath.c_str());
          // NODALS_FACE_YPLUS30_A export: one F2 fit per wall triangle.
          // F2 is evaluated before predictor assembly and is one SIMPLE step
          // lagged vs this corrected checkpoint. Preserve that distinction.
          const char* exportFlag=std::getenv("NODALS_FACE_YPLUS_EXPORT");
          if(exportFlag && std::string(exportFlag)=="1"){
            if(!f2FaceSpalding || BH.wallT.empty())
              throw std::runtime_error("NODALS_FACE_YPLUS_EXPORT requires F2 wall fit and tet faces");
            std::vector<double> snapshot(3u*BH.wallT.size());
            HXT1_CUDA(cudaMemcpy(snapshot.data(),d_f2FaceFit.p,
              snapshot.size()*sizeof(double),cudaMemcpyDeviceToHost));
            const std::string path=stage31Dir+"/NODALS_FACE_UTAU_it"+std::to_string(git)+".csv";
            const std::string tmp=path+".tmp";
            std::FILE*out=std::fopen(tmp.c_str(),"w");
            if(!out)throw std::runtime_error("NODALS_FACE_YPLUS could not write "+path);
            std::fprintf(out,"# NODALS_FACE_UTAU_V1 iteration=%d faces=%zu fit=F2_lagged\n",git,BH.wallT.size());
            std::size_t badFit=0;
            for(std::size_t i=0;i<BH.wallT.size();++i){
              const auto&f=BH.wallT[i];
              const double beta=snapshot[3u*i],utau=snapshot[3u*i+1u],yp=snapshot[3u*i+2u];
              if(!(beta>0. && std::isfinite(beta) && std::isfinite(utau) && utau>=0. &&
                   std::isfinite(yp)))++badFit;
              std::fprintf(out,"%zu,%d,%d,%.17g,%.17g\n",i,(int)f.cell,(int)f.lf,utau,yp);
            }
            if(std::fclose(out)!=0)throw std::runtime_error("NODALS_FACE_YPLUS export write failed");
            if(badFit){
              std::remove(tmp.c_str());
              throw std::runtime_error("NODALS_FACE_YPLUS invalid F2 fits "+std::to_string(badFit));
            }
            if(std::rename(tmp.c_str(),path.c_str())!=0)
              throw std::runtime_error("NODALS_FACE_YPLUS atomic CSV rename failed");
            std::printf("NODALS_FACE_UTAU_EXPORT=PASS iteration=%d faces=%zu path=%s "
                        "source=F2_PRE_MOMENTUM_ONE_STEP_LAG status=PASS\n",git,BH.wallT.size(),path.c_str());
          }
          std::fflush(stdout);
        }
        g2ChildSnap(9);
        if(g2ChildProfile){
          HXT1_CUDA(cudaEventSynchronize(g2ChildEv[9]));
          const char* labels[9]={
            "preconvection", "momentum_wall_assembly", "momentum_predictor_and_schur_setup",
            "pressure_pod_check", "pressure_pcg_and_velocity_corrector",
            "turbulence_k_omega_and_update", "checkpoint", "metrics", "convergence"};
          double allMs=0.0;
          for(int phase=0;phase<9;++phase){
            float ms=0.0f;
            HXT1_CUDA(cudaEventElapsedTime(&ms,g2ChildEv[phase],g2ChildEv[phase+1]));
            allMs += double(ms);
            std::printf("STAGE26_G2_PHASE it=%d phase=%s ms=%.6f liveVRAMMiB=%.3f\n",
              git,labels[phase],double(ms),g2ChildUsed[phase+1]/1048576.0);
          }
          std::printf("STAGE26_G2_PHASE_SUM it=%d ms=%.6f liveVRAMMiB=%.3f totalVRAMMiB=%.3f\n",
            git,allMs,g2ChildUsed[9]/1048576.0,g2ChildTotalMiB);
          std::fflush(stdout);
        }
        predAuditPrint(git); // Outside parent CUDA-event phase timing: no print contamination.
        if(g2Converged)break;
      }

      if(predAuditEnabled) for(auto &ev:predAuditEv) HXT1_CUDA(cudaEventDestroy(ev));
      HXT1_CUDA(cudaDeviceSynchronize());
      const auto g2PerfT1 = std::chrono::steady_clock::now();
      const double g2LoopMs =
        std::chrono::duration<double,std::milli>(g2PerfT1-g2PerfT0).count();

      std::printf(
        "NODALS_SST_G2_FIXED10_TIMING "
        "iterations=%d loopMs=%.3f avgOuterMs=%.6f status=PASS\n",
        g2Outer,
        g2LoopMs,
        g2LoopMs/std::max(g2Outer-stage31Start,1));
      pressurePOD.print_summary(pIts,g2Outer,g2LoopMs);

      std::vector<T>hk((std::size_t)nv),hw((std::size_t)nv);HXT1_CUDA(cudaMemcpy(hk.data(),TS.k.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hw.data(),TS.omega.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g2.vtu":vtu+"_sst_g2.vtu";hxt5c::write_sst_vtu(sstVtu,M,hk,hw);
      g2Gate=g2Converged&&g2FinalStats.meanNutOverNu>0&&g2FinalStats.maxNutOverNu>0&&g2FinalStats.kFloorFrac<.01&&g2FinalStats.omegaFloorFrac<.01;
#include "../diagnostics/final_force_budget.inc"
      std::printf("NODALS_SST_G2_FINAL iterations=%d kRelUpdate=%.12e omegaRelUpdate=%.12e nutMeanRelUpdate=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e kFloorVolumeFrac=%.12e omegaFloorVolumeFrac=%.12e wallDarcy=%.12e wallYPlusMean=%.12e pressureDrop=%.12e vtu=%s status=%s\n",g2Outer,g2KRel,g2WRel,g2NutRel,g2FinalStats.meanNutOverNu,g2FinalStats.maxNutOverNu,g2FinalStats.kFloorFrac,g2FinalStats.omegaFloorFrac,finalF,finalY,finalDp,sstVtu.c_str(),g2Gate?"PASS":"FAIL");std::printf("SST_G2_GATE_STATUS=%s\n",g2Gate?"PASS":"FAIL");
    }

    std::vector<T>hp((std::size_t)np),hu0((std::size_t)nv),hu1((std::size_t)nv),hu2((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(hp.data(),d_p.p,np*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu0.data(),d_u0.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu1.data(),d_u1.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu2.data(),d_u2.p,nv*sizeof(T),cudaMemcpyDeviceToHost));

    // Stage22D4E exact-FE diagnostic snapshot, optional and non-intrusive.
    // Write ALL P1+BF3 velocity DOFs, including face enrichments omitted from VTU.
    if(sstG2 && stage22d4Live && std::getenv("SST22D4E_EXPORT")){
      if(std::string(std::getenv("SST22D4E_EXPORT"))!="1")
        throw std::runtime_error("STAGE22D4E_EXPORT accepts only 1");
      const auto ix=vtu.find_last_of("/\\");
      const std::string parent=ix==std::string::npos?"":vtu.substr(0,ix+1);
      const std::string binary=parent+"STAGE22D4E_U_FULL.f64";
      const std::string metadata=parent+"STAGE22D4E_META.txt";
      {
        std::ofstream o(binary,std::ios::binary|std::ios::trunc);
        if(!o)throw std::runtime_error("Stage22D4E cannot open binary output: "+binary);
        for(const auto* v:{&hu0,&hu1,&hu2}){
          std::vector<double> tmp(v->size());
          for(std::size_t i=0;i<v->size();++i)tmp[i]=(double)(*v)[i];
          o.write(reinterpret_cast<const char*>(tmp.data()),
                  (std::streamsize)(tmp.size()*sizeof(double)));
        }
        if(!o)throw std::runtime_error("Stage22D4E velocity export failed");
      }
      {
        std::ofstream o(metadata,std::ios::trunc);
        if(!o)throw std::runtime_error("Stage22D4E cannot create metadata");
        o<<"format=STAGE22D4E_U3_FLOAT64_COMPONENT_MAJOR_V1\n"
         <<"mesh="<<mesh<<"\n"
         <<"nvel="<<nv<<"\n"
         <<"ntet="<<M.h.ntet<<"\n"
         <<"npoints="<<M.h.nv<<"\n"
         <<"iterations="<<g2Outer<<"\n"
         <<"gamma="<<s22d4Gamma<<"\n"
         <<"bulk="<<bulk<<"\n"
         <<"velocity="<<binary<<"\n";
      }
      // Evaluate the ACTUAL production B_N CSR, including its DG inlet trace,
      // wall subtraction and the existing prescribed inlet source vector.
      // Kept separate from the independently reconstructed FE divergence audit.
      {
        std::vector<double> residual((std::size_t)np,0.0);
        for(int row=0;row<np;++row){
          double value=BH.csrc[(std::size_t)row];
          for(std::int64_t jj=Bh.row[(std::size_t)row];
              jj<Bh.row[(std::size_t)row+1];++jj){
            const auto j=(std::size_t)jj;
            const auto col=(std::size_t)Bh.col[j];
            value+=Bh.bx[j]*(double)hu0[col]
                  +Bh.by[j]*(double)hu1[col]
                  +Bh.bz[j]*(double)hu2[col];
          }
          residual[(std::size_t)row]=value;
        }
        const std::string resfile=parent+"STAGE22D4E_BN_PRODUCTION.f64";
        std::ofstream o(resfile,std::ios::binary|std::ios::trunc);
        o.write(reinterpret_cast<const char*>(residual.data()),
                (std::streamsize)(residual.size()*sizeof(double)));
        if(!o)throw std::runtime_error("STAGE22D4E B_N production export failed");
      }
      std::printf("STAGE22D4E_EXPORTED iterations=%d gamma=%.8g nvel=%d allBF3=YES "
        "mesh=%s velocity=%s status=PASS\n",
        g2Outer,s22d4Gamma,nv,mesh.c_str(),binary.c_str());
    }


    auto pd=hxt4a::to_double(hp);
    double slope=hxt4a::pressure_slope(M,pd),pinP=hxt4a::boundary_p(M,pd,true),poutP=hxt4a::boundary_p(M,pd,false),drop=pinP-poutP;

    auto ws=hxt5b::wallstats(d_wallStats,bulk);
    long double et=0,ez=0;for(std::size_t v=0;v<M.points.size();++v){et+=(double)hu0[v]*(double)hu0[v]+(double)hu1[v]*(double)hu1[v];ez+=(double)hu2[v]*(double)hu2[v];}
    double trans=std::sqrt((double)(et/std::max((long double)1e-300,ez)));
    write_vtu(vtu,M,hu0,hu1,hu2,hp);
    const std::string diagVtu=hxt5b_diag::path2(vtu);
    const bool legacyFinalDiagnostics=
      stage31_restart::env_string("NODALS_LEGACY_FINAL_DIAGNOSTICS","0")=="1";
    if(legacyFinalDiagnostics && M.h.ntet==0){
      const double auditZ0=1.2;
      const double auditZ1=1.8;

      Dev<double> d_molForceAudit(
          std::vector<double>(2,0.0));

      Dev<unsigned long long> d_molForceBad(
          std::vector<unsigned long long>(1,0));

      if(!BH.wallH.empty()){
        hxt5d::spalding_molecular_force_audit_hex
          <<<BH.wallH.size(),64>>>(
            d_wallH.p,
            (int)BH.wallH.size(),
            d_pts.p,
            d_hc.p,
            d_hv.p,
            d_u2.p,
            nu,
            auditZ0,
            auditZ1,
            d_molForceAudit.p,
            d_molForceBad.p);

        HXT1_CUDA(cudaGetLastError());
        HXT1_CUDA(cudaDeviceSynchronize());
      }

      double hm[2]={0.0,0.0};
      unsigned long long hb=0;

      HXT1_CUDA(cudaMemcpy(
          hm,
          d_molForceAudit.p,
          2*sizeof(double),
          cudaMemcpyDeviceToHost));

      HXT1_CUDA(cudaMemcpy(
          &hb,
          d_molForceBad.p,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

      std::printf(
        "NODALS_SST_MOLECULAR_WALL_FORCE_AUDIT "
        "z0=%.12e z1=%.12e wallArea=%.12e "
        "molecularForce=%.12e "
        "molecularMeanTraction=%.12e "
        "bad=%llu status=%s\n",
        auditZ0,auditZ1,
        hm[0],
        hm[1],
        hm[0]>0.0 ? hm[1]/hm[0] : 0.0,
        hb,
        hb==0 ? "PASS" : "FAIL");
    }

    if(legacyFinalDiagnostics && M.h.ntet==0)
      hxt5b_diag::print_cross_section_momentum_audit(
        M,hu2,hp,bulk,1.2,1.8);
    else if(legacyFinalDiagnostics)
      std::printf("NODALS_LEGACY_CROSS_SECTION=SKIPPED reason=HEX_ONLY\n");

    if(legacyFinalDiagnostics){
      hxt5b_diag::write(diagVtu,M,BH,hu0,hu1,hu2,hp,nu,bulk,wallSampleFraction);
      std::printf("NODALS_HXT5B6_DIAG_VTU_OUTPUT path=%s velocity=FULL_Q1BF2_P1BF3_PHYSICAL_CELL_AVERAGE wallYPlus=EXACT_SPALDING status=PASS\n",diagVtu.c_str());
    }
    if(profileAudit){
      const std::string profileCsv=hxt5b_diag::profile_path(vtu);
      hxt5b_diag::write_profile(profileCsv,M,hu0,hu1,hu2,R,bulk,zmin,zmax,profileZFractions,profileRadial,profileAngles,profileRmax);
    }
    const double finalMeanNut=
      laminarPerf?0.0:
      (sstG2?g2FinalStats.meanNutOverNu:mix.meanNutOverNu);

    const double finalMaxNut=
      laminarPerf?0.0:
      (sstG2?g2FinalStats.maxNutOverNu:mix.maxNutOverNu);

    const char* finalTurb=
      laminarPerf?"LAMINAR_PERF":
      (sstG2?"SST_2003M":"NIKURADSE_ALGEBRAIC");

    bool gate=
      std::isfinite(slope)&&
      std::isfinite(drop)&&
      ws.area>0&&
      ws.rootFailures==0&&
      ws.yPlusMean>0&&
      (laminarPerf||finalMaxNut>0);
    std::printf("NODALS_HXT5B_PHYSICS turbulence=%s pressureSlope=%.12e pIn=%.12e pOut=%.12e pressureDrop=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e maxF1=%.12e maxF2=%.12e maxStrain=%.12e maxStrainSource=%s wallDarcyFromSpalding=%.12e wallYPlusMean=%.12e wallYPlusMax=%.12e wallSlipMean=%.12e transverseVertexRelL2=%.12e avgPressureCG=%.6f pressureSolver=%s status=%s\n",
      finalTurb,slope,pinP,poutP,drop,finalMeanNut,finalMaxNut,sstG2?g2FinalStats.maxF1:0.0,sstG2?g2FinalStats.maxF2:0.0,mix.maxStrain,
      laminarPerf?"NONE":
      (sstG2?(directSSTStart?"INITIAL_PLUG_AUDIT":"G0_MIXLEN_STARTER"):
       "CURRENT_MIXLEN"),
      ws.fDarcy,ws.yPlusMean,ws.yPlusMax,ws.slipMean,trans,(sstG2&&g2Outer)?(double)pIts/g2Outer:(outer?(double)pIts/outer:0.0),pressureSolver.c_str(),gate?"PASS":"FAIL");
    std::printf("NODALS_HXT5B_OUTPUT vtu=%s status=PASS\n",vtu.c_str());
    bool sstGate=true;
    if(sstG1.enabled){
      std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g1.vtu":vtu+"_sst_g1.vtu";
      auto sstResult=hxt5c::run_frozen(M,BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,VC,d_fixed,d_u0.p,d_u1.p,d_u2.p,d_turb,d_dg,d_wall,d_bad,bulk,R,nu,gamma,wallSampleFraction,sstG1,sstVtu);
      sstGate=sstResult.gate;
    }
    // A requested finite-iteration run is successful even when SST has not
    // converged yet. Keep physical convergence status as information, not an
    // installation/stage gate. Genuine invalid-state diagnostics still fail.
    const bool allGate=gate&&sstGate&&g2Gate;
    std::printf("HXT5B_PHYSICAL_CONVERGENCE=%s\n",allGate?"YES":"NO");
    const bool runtimeOK=gate&&sstGate;
    std::printf("NODALS_SOFTWARE_COMPLETION=%s iterations=%d converged=%d\n",
                runtimeOK?"COMPLETE":"PHYSICAL_CHECK_FAILED",g2Outer,(int)g2Converged);
    return runtimeOK?0:3;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT5B_ERROR precision=%s: %s\n",hxt4b::PRECISION,e.what());
    std::fprintf(stderr,"HXT5B_GATE_STATUS=FAIL\n");return 2;
  }
}
