#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf $SCRIPT_DIR/build/*.ncu-rep

ncu --config-file off --export "$SCRIPT_DIR/build/SimpleCsrSpMM" --import-source yes --force-overwrite --set full $SCRIPT_DIR/build/SimpleCsrSpMMTest
ncu --config-file off --export "$SCRIPT_DIR/build/CRCSpMM" --import-source yes --force-overwrite --set full $SCRIPT_DIR/build/CRCSpMMTest
ncu --config-file off --export "$SCRIPT_DIR/build/CRC-CWM-SpMM" --import-source yes --force-overwrite --set full $SCRIPT_DIR/build/CRC-CWM-SpMM-Test