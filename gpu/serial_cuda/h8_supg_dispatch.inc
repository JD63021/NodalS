int main(int argc,char**argv)
{
  const H8SupgCli q=h8_supg_scan(argc,argv);
  if(!q.enabled)return h8_base_without_supg_cli(argc,argv);
  return h8_supg_main(argc,argv);
}
