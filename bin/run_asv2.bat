@echo off
set current=%~dp0
cd %~dp0..\_benches
rem goto publish:
@echo --CREATE-BENCH--
python -m mlprodict asv_bench --location . --models "LinearRegression" --build "../build"
@echo --RUN-BENCH--
python -m asv run --show-stderr --config=asv.conf.json --environment existing:same
:publish:
@echo --PUBLISH--
if exist ..\build\html rmdir ..\build\html -r -f
python -m asv publish -o ..\build\html
@echo --END--
set %current%