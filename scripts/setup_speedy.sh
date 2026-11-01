#!/bin/bash
# Stage 14/15: build the patched SPEEDY (speedy.f90) used as the simulator environment and teacher
# (&control tend_out = .true. also writes the per-process physics tendencies, yyyymmddhhmm_tend.nc).
#   bash scripts/setup_speedy.sh DEST      (DEST/bin/speedy, DEST/data/bc/t30 afterwards)
# Needs gfortran and the NetCDF Fortran library (Ubuntu: apt-get install gfortran libnetcdff-dev).
set -e
DEST=${1:?usage: setup_speedy.sh DEST}
REPO=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=0ebfdbfcd62efd46dd1802284908645be4b230be
if ! command -v gfortran > /dev/null || [ ! -f /usr/include/netcdf.mod ]; then
  apt-get install -y gfortran libnetcdff-dev
fi
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/samhatfield/speedy.f90 "$DEST"
fi
cd "$DEST"
git checkout -q $COMMIT
if ! grep -q env_control source/gfortran.makefile; then
  git apply "$REPO/src/climate_wm/sim/speedy_env.patch"
fi
NETCDF=${NETCDF:-/usr} bash build.sh
echo "built $DEST/bin/speedy"
