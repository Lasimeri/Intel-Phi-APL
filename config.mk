# config.mk: the paths the APL engine builds against. Every value can be
# overridden on the command line (make PHI_CARD=1 check) or from the
# environment; scripts/lib.sh applies the same defaults when a script runs
# on its own. PHI_TREE contains spaces: it is only ever passed to scripts,
# never used as a make prerequisite.
PHI_TREE          ?= /media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A
PHI_LLVM_BIN      ?= $(HOME)/.cache/intel-phi-llvm-build/llvm/bin
PHI_SYSROOT_7120A ?= $(HOME)/.cache/intel-phi-7120a-build/toolchain/sysroot
PHI_CARD          ?= 0
HOST_CC           ?= cc
LLAMA_DIR         ?= $(HOME)/llama.cpp

export PHI_TREE PHI_LLVM_BIN PHI_SYSROOT_7120A PHI_CARD HOST_CC LLAMA_DIR
