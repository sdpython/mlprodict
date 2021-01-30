@echo off
set current=%~dp0
cd %~dp0..
echo --MACHINE--
python -m asv machine --yes --config mlprodict/asv_benchmark/asv.conf.json
echo .
echo --BENCH--
python -m asv run --show-stderr --config mlprodict/asv_benchmark/asv.conf.json --environment existing:same
echo --CLEAN--
if exist dist/html/asv rmdir dist/html/asv -r -f
rem if exist build/env rmdir build/env -r -f

echo --PUBLISH--
python -m asv publish --config mlprodict/asv_benchmark/asv.conf.json -o ../../dist/asv/html

echo --END--
cd %current%