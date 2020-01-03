@echo off
set current=%~dp0
cd %~dp0..
rem goto publish:
@echo --CLEAN--
if exist build\html rmdir build\html -r -f
@echo --CREATEBENCH--
python -m mlprodict asv_bench --location _benches --models "StandardScaler" --build "build"
@echo --RUNBENCH--
python -m asv run --show-stderr --config=_benches\\asv.conf.json -v
:publish:
@echo --PUBLISH--
python -m asv publish -o build\html --config=_benches\\asv.conf.json
@echo --END--
cd %current%