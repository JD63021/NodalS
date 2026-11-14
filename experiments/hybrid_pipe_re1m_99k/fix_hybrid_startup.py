#!/usr/bin/env python3
"""Allow the original mixed HEX/TET SST path through pure-tet-only startup.
Tiny, idempotent source patch: one existing C++ app .inc and Python runner.
Preserves the untouched pure-tetra startup for all pre-existing cases.
"""
from pathlib import Path
import argparse

def replace_once(path, before, after):
    data=path.read_text()
    if after in data:
        print("ALREADY_PATCHED",path)
        return
    count=data.count(before)
    if count!=1:
        raise RuntimeError(f"{path}: expected exactly one anchor, got {count}: {before[:90]!r}")
    path.write_text(data.replace(before,after,1))
    print("PATCHED",path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--app",type=Path,required=True)
    a=ap.parse_args()
    app=a.app.expanduser().resolve()
    py=app/"nodals.py"
    cpp=app/"source/puretet_profile5_eq5_hxt5b/modules/app/sst_application.inc"
    if not py.is_file() or not cpp.is_file():
        raise FileNotFoundError("missing NodalS SST source under "+str(app))

    # The native parser checks the *presence* of SST22D4_NITSCHE_ENABLE,
    # rejecting '0'.  Omit it entirely in legacy hybrid mode.
    replace_once(py,
        "            'SST22D4_NITSCHE_ENABLE':'0',\n",
        "            # SST22D4_NITSCHE_ENABLE is absent for legacy HEX/TET.\n")

    replace_once(py,
        "        })\\n    env['SST_G2_VTU_DIR']",
        "        })\\n        env.pop('SST22D4_NITSCHE_ENABLE', None)\\n    env['SST_G2_VTU_DIR']")

    replace_once(cpp,
        """    nodals::stage25::rotate_loaded_tets(M,stage25Degrees);
    nodals::mesh::route_operator_patches(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");
    const auto stage19Patches=nodals::mesh::build_registry(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");""",
        """    // Original Stage25 rotation, Stage19 tetra-only registry and Stage23C
    // arbitrary patch routing are pure-tet facilities.  The straight hybrid
    // pipe already carries native HXT1 HEX/TET patch records; keep them.
    nodals::mesh::Registry stage19Patches;
    if(M.h.nhex>0){
      if(M.h.ntet==0 || M.h.ninterface==0 ||
         std::abs(stage25Degrees)>1.e-14)
        throw std::runtime_error("HYBRID_INIT needs HEX+TET interface and zero rotation");
      const char* patchMap=std::getenv("SST_PATCH_MAP_FILE");
      if(patchMap && *patchMap)
        throw std::runtime_error("HYBRID_INIT uses HXT1 native patches; custom patch_map unsupported");
      nodals::stage25::set_angle(0.);
      std::printf("NODALS_HYBRID_INIT zeroRotation=YES nativeHXT1Patches=YES hex=%llu tet=%llu interfaces=%llu status=PASS\\n",
        (unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,
        (unsigned long long)M.h.ninterface);
    }else{
      nodals::stage25::rotate_loaded_tets(M,stage25Degrees);
      nodals::mesh::route_operator_patches(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");
      stage19Patches=nodals::mesh::build_registry(M,std::getenv("SST_PATCH_MAP_FILE")?std::getenv("SST_PATCH_MAP_FILE"):"");
    }""")

    replace_once(cpp,
        "    nodals::mesh::audit_legacy_operators(stage19Patches,BH,M);",
        """    if(M.h.nhex==0){
      nodals::mesh::audit_legacy_operators(stage19Patches,BH,M);
    }else{
      if(BH.inH.size()!=M.inHex.size() || BH.inT.size()!=M.inTet.size() ||
         BH.wallH.size()!=M.wallHex.size() || !BH.wallT.empty() ||
         BH.wallH.empty() || !(BH.inletArea>0.0) ||
         !(BH.sourceSum<0.0))
        throw std::runtime_error("HYBRID_INIT inlet/wall boundary plan mismatch");
      std::printf("NODALS_HYBRID_BOUNDARY hexIn=%zu tetIn=%zu hexWall=%zu "
        "inletArea=%.12e inletSource=%.12e continuityAudit=HXT5B_NATIVE status=PASS\\n",
        BH.inH.size(),BH.inT.size(),BH.wallH.size(),
        BH.inletArea,BH.sourceSum);
    }""")

    print("HYBRID_NATIVE_STARTUP_PATCH=PASS; single CUDA translation-unit rebuild needed")

if __name__=="__main__":
    main()
