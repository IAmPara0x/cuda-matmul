{
  description = "A very basic flake";

  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/cbc4211f0afffe6dfd2478a62615dd5175a13f9a";
    };
  };

  outputs = { self, nixpkgs }: 

  let 
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config = { allowUnfree = true; }; };
    cudatoolkit = pkgs.cudaPackagesGoogle.cudatoolkit;

    # buildScript = 
    #
    # /* bash */
    # ''
    #
    #   set -xe
    #   outfile=$(basename "$1" .cu)
    #   nvcc ./matrix.cpp "$1" -o "$outfile" -I ${cudatoolkit}/include -ldir ${cudatoolkit}/nvvm/libdevice/ -L ${cudatoolkit}/lib -L ${cudatoolkit.lib}/lib  --dont-use-profile -G -rdc=true -lcudadevrt
    #   patchelf --set-rpath "/run/opengl-driver/lib:"$(patchelf --print-rpath "$outfile") "$outfile"
    # '';
    #
    # cudaCompile = pkgs.writeScriptBin "cudaCompile" 
    # ''
    #   #!${pkgs.stdenv.shell}
    #   ${buildScript}
    # '';
    #
    shell = pkgs.mkShell.override { stdenv = pkgs.gcc11Stdenv; } {
      packages = with pkgs; [gcc11 gcc11Stdenv cudatoolkit hyperfine];
      CUDA_TOOLKIT = "${cudatoolkit}";
      CUDA_TOOLKIT_LIB = "${cudatoolkit.lib}";
    };
  in
  {
   devShells.${system}.default = shell;
  };
}
