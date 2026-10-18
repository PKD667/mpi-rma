{
  description = "mpi-rma, and its benchmarks as nixsci experiments";

  inputs = {
    nixsci.url = "github:PKD667/nixsci/v0.1.0";
    nixpkgs.follows = "nixsci/nixpkgs";
  };

  outputs = { self, nixpkgs, nixsci }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      l = nixsci.lib.lab { inherit pkgs; project = self; };
      mpi = pkgs.openmpi;
      python = nixsci.packages.${system}.python;

      # The three example binaries the benchmarks run. The source set leaves out data, docs, lab.lock and
      # nixsci/, so recording a run or editing a script does not change the closure the runs name.
      examples = pkgs.rustPlatform.buildRustPackage {
        pname = "mpi-rma-examples";
        version = "0.1.0";
        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [ ./Cargo.toml ./Cargo.lock ./src ./examples ./tests ./README.md ];
        };
        cargoLock.lockFile = ./Cargo.lock;
        nativeBuildInputs = [ pkgs.pkg-config pkgs.rustPlatform.bindgenHook mpi ];
        buildInputs = [ mpi ];
        MPICC = "${mpi}/bin/mpicc";
        cargoBuildFlags = [ "--examples" ];
        doCheck = false;
        installPhase = ''
          mkdir -p $out/bin
          for e in compare soak; do
            cp target/${pkgs.stdenv.hostPlatform.rust.rustcTarget}/release/examples/$e $out/bin/
          done
        '';
      };

      data = import ./nixsci/data.nix;

      # One closure per benchmark: the same examples, a different `bench.py` case.
      closure = name: nixsci.lib.mkExperiment pkgs {
        name = "mpi-rma-${name}";
        program = "${python}/bin/python";
        args = [ "${./nixsci/bench.py}" name ];
        runtime = [ mpi ];
        env = {
          MPI_RMA_EXAMPLES = "${examples}/bin";
          # Open MPI 5 selects its UCX one-sided component and fails on a host without a UCX transport.
          # Excluding it leaves the shared-memory component the benchmarks are about.
          OMPI_MCA_osc = "^ucx";
        };
      };

      experiment = name: params: dataset: l.experiment {
        name = "mpi-rma-${name}";
        closures.${system} = closure name;
        inherit params;
        data.${dataset} = data.${dataset};
      };

      # Ranks: this repository's published numbers (BENCHMARKS.md) used 24 on a dual-Xeon node. The
      # default here is one rank per hardware thread of a 12-thread laptop, and the run's machine.json
      # records what it actually was.
      ranks = 12;
    in
    {
      packages.${system} = { inherit examples; };

      lab.${system} = rec {
        experiments = {
          compare = experiment "compare" { inherit ranks; repeats = 11; messages = 20000; depth = 32; } "compare";
          loss = experiment "loss" { inherit ranks; repeats = 5; messages = 20000; } "soak";
          scale = experiment "scale" { inherit ranks; repeats = 5; messages = 20000; } "soak";
        };
      };

      apps.${system} = lib.mapAttrs (_: e: e.run) self.lab.${system}.experiments;
    };
}
