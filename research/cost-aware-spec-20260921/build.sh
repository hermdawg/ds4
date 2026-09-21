#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
make -j8 ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o
# Keep finite/NaN checks real: production's -ffast-math is unsuitable for
# validating externally supplied measurements. This does not rebuild ds4.
cc -O3 -g -mcpu=native -Wall -Wextra -Wno-unused-function -std=c99 -I. \
  -dynamiclib research/cost-aware-spec-20260921/bridge.c \
  ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o \
  ds4_layer_pack.o ds4_engram.o -lm -pthread -framework Foundation -framework Metal \
  -o research/cost-aware-spec-20260921/bridge.dylib
