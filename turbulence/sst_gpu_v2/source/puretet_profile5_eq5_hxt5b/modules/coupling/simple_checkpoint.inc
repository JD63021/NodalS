// Restored missing G2 visualization checkpoint block.
// This is included INSIDE the SIMPLE iteration, after k/omega updates.
// It writes the unchanged BF3 momentum and vertex-only SST visualization.
// It does NOT provide a P2 scalar restart, nor does it activate P2 SST.
if (sstG2VtuEvery > 0 && git % sstG2VtuEvery == 0) {
  const std::string checkpointStem =
      sstG2VtuDir + (sstG2VtuDir.empty() || sstG2VtuDir.back() == '/' ? "" : "/") +
      "NODALS_SST_G2_CHECKPOINT_it" + std::to_string(git);
  const std::string flowCheckpoint = checkpointStem + "_flow.vtu";
  const std::string sstCheckpoint = checkpointStem + "_sst.vtu";
  std::vector<T> checkpointU0((std::size_t)nv);
  std::vector<T> checkpointU1((std::size_t)nv);
  std::vector<T> checkpointU2((std::size_t)nv);
  std::vector<T> checkpointP((std::size_t)np);
  std::vector<T> checkpointK((std::size_t)nv);
  std::vector<T> checkpointOmega((std::size_t)nv);
  HXT1_CUDA(cudaMemcpy(checkpointU0.data(), d_u0.p, (std::size_t)nv * sizeof(T), cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(checkpointU1.data(), d_u1.p, (std::size_t)nv * sizeof(T), cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(checkpointU2.data(), d_u2.p, (std::size_t)nv * sizeof(T), cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(checkpointP.data(), d_p.p, (std::size_t)np * sizeof(T), cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(checkpointK.data(), TS.k.p, (std::size_t)nv * sizeof(T), cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(checkpointOmega.data(), TS.omega.p, (std::size_t)nv * sizeof(T), cudaMemcpyDeviceToHost));
  write_vtu(flowCheckpoint, M, checkpointU0, checkpointU1, checkpointU2, checkpointP);
  hxt5c::write_sst_vtu(sstCheckpoint, M, checkpointK, checkpointOmega);
  std::printf("NODALS_SST_G2_VTU_CHECKPOINT it=%d flow=%s sst=%s gateIndependent=1 status=PASS\n",
              git, flowCheckpoint.c_str(), sstCheckpoint.c_str());
  std::fflush(stdout);
}
