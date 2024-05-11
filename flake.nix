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

    shell = pkgs.mkShell.override { stdenv = pkgs.gcc11Stdenv; } {
      packages = with pkgs; [gcc11 gcc11Stdenv cudatoolkit hyperfine valgrind];
      CUDA_TOOLKIT = "${cudatoolkit}";
      CUDA_TOOLKIT_LIB = "${cudatoolkit.lib}";
    };
  in
  {
   devShells.${system}.default = shell;
  };
}
