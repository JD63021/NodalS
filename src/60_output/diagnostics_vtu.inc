static PetscErrorCode computeErrorsRoot(const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]; for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d]; auto Q=tetDuffy7(); PetscInt nv=M.points.size();
  double total=0,intdiff=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);for(auto&q:Q){double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double w=q.w*det;total+=w;intdiff+=(p[(std::size_t)D.pGid[c]]-exactP(x,y,z))*w;}}
  double shift=intdiff/total, ue2=0,un2=0,pe2=0,pn2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);PetscInt lg[8];for(int i=0;i<4;++i)lg[i]=t[i];for(int i=0;i<4;++i)lg[4+i]=nv+M.oppFace[c][i];double coeff[3][8]={{0}};for(int a=0;a<8;++a){PetscInt ia=D.g2free[lg[a]];if(ia>=0)for(int d=0;d<3;++d)coeff[d][a]=(*ua[d])[(std::size_t)ia];}
    for(auto&q:Q){double val[8],gr[8][3];basis(q.lam,val,gr);double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double uh[3]={0},ue[3];for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];exactU(x,y,z,ue);double pp=exactP(x,y,z),w=q.w*det;for(int d=0;d<3;++d){ue2+=(uh[d]-ue[d])*(uh[d]-ue[d])*w;un2+=ue[d]*ue[d]*w;}double de=p[(std::size_t)D.pGid[c]]-shift-pp;pe2+=de*de*w;pn2+=pp*pp*w;}}
PetscCall(PetscPrintf(PETSC_COMM_SELF,"P1BF3_MMS U_L2=%.12e U_relL2=%.12e P_shifted_L2=%.12e P_shifted_relL2=%.12e pressureShift=%.12e\n",std::sqrt(ue2),std::sqrt(ue2/un2),std::sqrt(pe2),std::sqrt(pe2/pn2),shift)); PetscFunctionReturn(PETSC_SUCCESS);
}

static double gatheredEntityValue(const Discrete& D,const std::vector<double>& ua,int d,PetscInt entity) {
  const PetscInt gid=D.g2free[entity]; return gid>=0?ua[(std::size_t)gid]:entityDirValue(D,d,entity);
}

static void fixedFaceMean(const Mesh& M,const Discrete& D,PetscInt f,double avg[3]) {
  const PetscInt nv=(PetscInt)M.points.size(); const auto& F=M.faces[f];
  for(int d=0;d<3;++d) {
    avg[d]=(entityDirValue(D,d,F.v[0])+entityDirValue(D,d,F.v[1])+entityDirValue(D,d,F.v[2]))/3.0;
    avg[d]+=(9.0/20.0)*entityDirValue(D,d,nv+f);
  }
}

static PetscErrorCode auditFixedNormalInletRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P) {
  PetscFunctionBeginUser;
  if(P.inletBC!=InletBCMode::FixedNormalSpeed) PetscFunctionReturn(PETSC_SUCCESS);
  const auto& B=P.boundary; const auto& pin=M.patches[B.inlet];
  double flux=0.0,unMin=std::numeric_limits<double>::max(),unMax=-std::numeric_limits<double>::max();
  double meanVectorErrMax=0.0;
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) {
    const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]; fixedFaceMean(M,D,f,avg); const Vec3 u{avg[0],avg[1],avg[2]};
    flux += dot3(u,sf); const double un=dot3(u,nf); unMin=std::min(unMin,un); unMax=std::max(unMax,un);
    meanVectorErrMax=std::max(meanVectorErrMax,norm3(sub3(u,B.inletVelocity)));
  }
  const double target=B.signedNormalSpeed*B.inletProjectedArea;
  const double relErr=std::abs(flux-target)/std::max(std::abs(target),1e-300);
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_INLET_BC_AUDIT patch=%s mode=fixed_normal_speed normalMode=average_patch_normal signedSpeed=%.12e referenceNormal=[%.12e,%.12e,%.12e] U=[%.12e,%.12e,%.12e] area=%.12e projectedArea=%.12e planarityRatio=%.12e targetFlux=%.12e imposedFlux=%.12e relFluxError=%.12e faceNormalVelocityMin=%.12e faceNormalVelocityMax=%.12e faceMeanVectorErrorMax=%.12e status=%s\n",
    B.inletPatch.c_str(),B.signedNormalSpeed,B.inletReferenceNormal.x,B.inletReferenceNormal.y,B.inletReferenceNormal.z,
    B.inletVelocity.x,B.inletVelocity.y,B.inletVelocity.z,B.inletArea,B.inletProjectedArea,B.inletProjectedArea/B.inletArea,
    target,flux,relErr,unMin,unMax,meanVectorErrMax,(relErr<1e-12 && meanVectorErrMax<1e-12)?"PASS":"CHECK"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct PatchSolutionStats {
  double area=0.0,flux=0.0,pOwnerMean=0.0,normalVelocityMin=std::numeric_limits<double>::max(),normalVelocityMax=-std::numeric_limits<double>::max();
};

static PatchSolutionStats patchSolutionStatsRoot(const Mesh& M,const Discrete& D,int pi,const std::vector<double>* ua[3],const double* pa) {
  PatchSolutionStats S; const PetscInt nv=(PetscInt)M.points.size(); const auto& P=M.patches[pi]; double psum=0.0;
  for(PetscInt f=P.startFace;f<P.startFace+P.nFaces;++f) {
    const auto& F=M.faces[f]; const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]={0,0,0};
    for(int d=0;d<3;++d) {
      avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
      avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
    }
    const Vec3 u{avg[0],avg[1],avg[2]}; const double un=dot3(u,nf);
    S.area+=area; S.flux+=dot3(u,sf); psum+=area*pa[D.pGid[M.owner[f]]];
    S.normalVelocityMin=std::min(S.normalVelocityMin,un); S.normalVelocityMax=std::max(S.normalVelocityMax,un);
  }
  S.pOwnerMean=psum/std::max(S.area,1e-300); return S;
}

static PetscErrorCode computeFlowDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const auto in=patchSolutionStatsRoot(M,D,P.boundary.inlet,ua,p.data()),out=patchSolutionStatsRoot(M,D,P.boundary.outlet,ua,p.data());
  const double net=in.flux+out.flux,scale=std::max({std::abs(in.flux),std::abs(out.flux),1e-300});
  const double target=(P.inletBC==InletBCMode::FixedNormalSpeed)?P.boundary.signedNormalSpeed*P.boundary.inletProjectedArea:0.0;
  const double targetRel=(P.inletBC==InletBCMode::FixedNormalSpeed)?std::abs(in.flux-target)/std::max(std::abs(target),1e-300):0.0;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_DIAGNOSTICS inlet=%s outlet=%s inletFlux=%.12e targetInletFlux=%.12e inletFluxRelError=%.12e outletFlux=%.12e netOutwardFlux=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e inletFaceNormalVelocity=[%.12e,%.12e] outletFaceNormalVelocity=[%.12e,%.12e]\n",
    P.boundary.inletPatch.c_str(),P.boundary.outletPatch.c_str(),in.flux,target,targetRel,out.flux,net,std::abs(net)/scale,in.area,out.area,
    in.normalVelocityMin,in.normalVelocityMax,out.normalVelocityMin,out.normalVelocityMax));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropInMinusOut=%.12e pressureGauge=physical_outlet_no_nullspace\n",
    in.pOwnerMean,out.pOwnerMean,in.pOwnerMean-out.pOwnerMean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static Vec3 tetCellAverageVelocityRoot(const Mesh& M,const Discrete& D,PetscInt c,const std::vector<double>* ua[3]) {
  // Exact volume mean for P1+BF3 on an affine tetrahedron:
  // mean(lambda_i)=1/4 and mean(27*lambda_j*lambda_k*lambda_l)=9/40.
  const PetscInt nv=(PetscInt)M.points.size();
  PetscInt ent[8];
  for(int i=0;i<4;++i) ent[i]=M.tets[c][i];
  for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[c][i];
  double u[3]={0.0,0.0,0.0};
  for(int d=0;d<3;++d) {
    for(int i=0;i<4;++i) u[d]+=0.25*gatheredEntityValue(D,*ua[d],d,ent[i]);
    for(int i=0;i<4;++i) u[d]+=(9.0/40.0)*gatheredEntityValue(D,*ua[d],d,ent[4+i]);
  }
  return Vec3{u[0],u[1],u[2]};
}

static double tetVolumeAbsRoot(const Mesh& M,PetscInt c) {
  const auto& t=M.tets[c];
  const Vec3& X0=M.points[t[0]]; const Vec3& X1=M.points[t[1]];
  const Vec3& X2=M.points[t[2]]; const Vec3& X3=M.points[t[3]];
  double J[3][3]={{X1.x-X0.x,X2.x-X0.x,X3.x-X0.x},
                  {X1.y-X0.y,X2.y-X0.y,X3.y-X0.y},
                  {X1.z-X0.z,X2.z-X0.z,X3.z-X0.z}};
  return std::abs(det3(J))/6.0;
}

static PetscErrorCode writeVtuRoot(const std::string& path,const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p,PetscBool converged,PetscInt outerIts,const std::string& velocityMode) {
  PetscFunctionBeginUser;
  const bool writeLegacy=(velocityMode=="legacy" || velocityMode=="both");
  const bool writeU0=(velocityMode=="cell_average" || velocityMode=="u0" || velocityMode=="both");
  if(!writeLegacy && !writeU0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"unknown VTU velocity mode %s",velocityMode.c_str());

  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];

  // U0 is the exact element-volume average of the solved P1+BF3 polynomial.
  // U0_stream is a volume-weighted vertex average of neighboring U0 cells.  It
  // exists only as a continuous visualization surrogate for ParaView streamlines.
  std::vector<Vec3> u0, u0Stream;
  if(writeU0) {
    u0.resize(M.tets.size());
    u0Stream.assign(M.points.size(),Vec3{0.0,0.0,0.0});
    std::vector<double> w(M.points.size(),0.0);
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
      u0[c]=tetCellAverageVelocityRoot(M,D,c,ua);
      const double vol=tetVolumeAbsRoot(M,c);
      for(int i=0;i<4;++i) {
        const PetscInt v=M.tets[c][i];
        u0Stream[v].x+=vol*u0[c].x; u0Stream[v].y+=vol*u0[c].y; u0Stream[v].z+=vol*u0[c].z;
        w[v]+=vol;
      }
    }
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) if(w[v]>0.0) {
      u0Stream[v].x/=w[v]; u0Stream[v].y/=w[v]; u0Stream[v].z/=w[v];
    }
  }

  std::ofstream out(path); if(!out) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"cannot open VTU output %s",path.c_str());
  out<<std::setprecision(17)<<std::scientific;
  out<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n";
  out<<"<Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<M.tets.size()<<"\">\n";
  out<<"<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto& x:M.points) out<<x.x<<' '<<x.y<<' '<<x.z<<'\n'; out<<"</DataArray></Points>\n";
  out<<"<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for(const auto& t:M.tets) out<<t[0]<<' '<<t[1]<<' '<<t[2]<<' '<<t[3]<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<4*(c+1)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<10<<'\n'; out<<"</DataArray>\n</Cells>\n";

  out<<"<PointData";
  if(writeU0) out<<" Vectors=\"U0_stream\""; else if(writeLegacy) out<<" Vectors=\"U_P1\"";
  out<<">\n";
  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) out<<gatheredEntityValue(D,*ua[0],0,v)<<' '<<gatheredEntityValue(D,*ua[1],1,v)<<' '<<gatheredEntityValue(D,*ua[2],2,v)<<'\n';
    out<<"</DataArray>\n";
  }
  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0_stream\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0Stream) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }
  out<<"</PointData>\n";

  out<<"<CellData Scalars=\"p_P0\" Vectors=\""<<(writeU0?"U0":"U_P1BF3_centroid")<<"\">\n";
  out<<"<DataArray type=\"Float64\" Name=\"p_P0\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) out<<p[(std::size_t)D.pGid[c]]<<'\n'; out<<"</DataArray>\n";

  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }

  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { double uc[3]={0,0,0}; for(int i=0;i<4;++i) for(int d=0;d<3;++d) uc[d]+=0.25*gatheredEntityValue(D,*ua[d],d,M.tets[c][i]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
    out<<"<DataArray type=\"Float64\" Name=\"U_P1BF3_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    std::array<double,4> lam{{0.25,0.25,0.25,0.25}}; double val[8],gr[8][3]; basis(lam,val,gr); const PetscInt nv=(PetscInt)M.points.size();
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[c][i]; double uc[3]={0,0,0}; for(int a=0;a<8;++a) for(int d=0;d<3;++d) uc[d]+=val[a]*gatheredEntityValue(D,*ua[d],d,ent[a]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
  }

  out<<"<DataArray type=\"Int32\" Name=\"solve_converged\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<(converged?1:0)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int32\" Name=\"outer_iterations\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<outerIts<<'\n'; out<<"</DataArray>\n";
  out<<"</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; out.close();

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_VTU path=%s mode=%s points=%zu cells=%zu pointVelocity=%s cellVelocity=%s cellPressure=p_P0 converged=%d outerIts=%" PetscInt_FMT " status=PASS\n",
    path.c_str(),velocityMode.c_str(),M.points.size(),M.tets.size(),writeU0?"U0_stream":"U_P1",writeU0?"U0":"U_P1BF3_centroid",converged?1:0,outerIts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computePipeDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const auto& G=P.pipe;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const PetscInt nv=(PetscInt)M.points.size();
  const auto Q=tetDuffy7();
  double ue2=0,un2=0,ueMesh2=0,trans2=0,pe2=0,pn2=0,total=0,pdiff=0;
  double sw=0,sz=0,szz=0,sp=0,szp=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),vol=det/6.0;
    PetscInt entity[8];for(int i=0;i<4;++i)entity[i]=t[i];for(int i=0;i<4;++i)entity[4+i]=nv+M.oppFace[c][i];
    double coeff[3][8]={{0}};for(int a=0;a<8;++a)for(int d=0;d<3;++d)coeff[d][a]=gatheredEntityValue(D,*ua[d],d,entity[a]);
    const double pc=p[(std::size_t)D.pGid[c]];
    double zc=0;for(int i=0;i<4;++i)zc+=X[i].z;zc*=0.25;
    sw+=vol;sz+=vol*zc;szz+=vol*zc*zc;sp+=vol*pc;szp+=vol*zc*pc;
    for(const auto& q:Q) {
      double val[8],gr[8][3];basis(q.lam,val,gr);
      double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}
      double uh[3]={0,0,0};for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];
      const double uz=pipeIdealUz(G,x,y),uzMesh=G.profileScale*uz,pex=pipeExactPressure(G,z),w=q.w*det;
      ue2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uz)*(uh[2]-uz))*w;
      ueMesh2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uzMesh)*(uh[2]-uzMesh))*w;
      un2+=uz*uz*w;trans2+=(uh[0]*uh[0]+uh[1]*uh[1])*w;
      const double dp=pc-pex;pe2+=dp*dp*w;pn2+=pex*pex*w;pdiff+=dp*w;total+=w;
    }
  }
  const double pshift=pdiff/total;
  double peShift2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),pc=p[(std::size_t)D.pGid[c]];
    for(const auto& q:Q){double z=0;for(int i=0;i<4;++i)z+=q.lam[i]*X[i].z;const double de=pc-pshift-pipeExactPressure(G,z);peShift2+=de*de*q.w*det;}
  }

  auto faceFluxAndPressure=[&](int pi,double& flux,double& pmean) {
    flux=0.0;pmean=0.0;double areaSum=0.0;const auto& pp=M.patches[pi];
    for(PetscInt f=pp.startFace;f<pp.startFace+pp.nFaces;++f) {
      const auto& F=M.faces[f];const Vec3& x0=M.points[F.v[0]];const Vec3& x1=M.points[F.v[1]];const Vec3& x2=M.points[F.v[2]];
      const double area=triangleArea(x0,x1,x2);double avg[3]={0,0,0};
      for(int d=0;d<3;++d) {
        avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
        avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
      }
      flux+=area*avg[2];
      pmean+=area*p[(std::size_t)D.pGid[M.owner[f]]];areaSum+=area;
    }
    pmean/=areaSum;
  };
  double qIn=0,pIn=0,qOut=0,pOut=0;faceFluxAndPressure(G.inlet,qIn,pIn);faceFluxAndPressure(G.outlet,qOut,pOut);
  const double massRel=(qIn!=0)?(qOut-qIn)/qIn:0.0;
  const double denom=sw*szz-sz*sz;
  const double slope=(std::abs(denom)>1e-300)?(sw*szp-sz*sp)/denom:0.0;
  const double fitDrop=-slope*G.L,faceDrop=pIn-pOut;

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_ERROR U_L2=%.12e U_relL2=%.12e U_meshNormalized_L2=%.12e U_meshNormalized_relL2=%.12e transverse_L2=%.12e P_abs_L2=%.12e P_abs_relL2=%.12e P_shifted_L2=%.12e pressureShift=%.12e\n",
    std::sqrt(ue2),std::sqrt(ue2/std::max(un2,1e-300)),std::sqrt(ueMesh2),std::sqrt(ueMesh2/std::max(G.profileScale*G.profileScale*un2,1e-300)),std::sqrt(trans2),std::sqrt(pe2),std::sqrt(pe2/std::max(pn2,1e-300)),std::sqrt(peShift2),pshift));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_FLOW inletFlux=%.12e outletFlux=%.12e inletMean=%.12e outletMean=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e\n",
    qIn,qOut,qIn/G.inletArea,qOut/G.outletArea,massRel,G.inletArea,G.outletArea));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropFaceOwner=%.12e dropAxialFit=%.12e hpDrop=%.12e faceRelError=%.12e fitRelError=%.12e\n",
    pIn,pOut,faceDrop,fitDrop,G.hpDrop,(faceDrop-G.hpDrop)/G.hpDrop,(fitDrop-G.hpDrop)/G.hpDrop));

 
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyDiscrete(Discrete& D){PetscFunctionBeginUser;PetscCall(MatDestroy(&D.A));for(int d=0;d<3;++d){PetscCall(MatDestroy(&D.B[d]));PetscCall(VecDestroy(&D.rhs[d]));}PetscCall(VecDestroy(&D.volumes));PetscCall(VecDestroy(&D.fixedDiv));PetscFunctionReturn(PETSC_SUCCESS);}

