@echo off
@echo --RUN-BENCH--
set current=%~dp0
cd %~dp0..\_benches
python -m asv run --show-stderr --config=asv.conf.json
if exist ..\build\html rm ..\build\html -r -f
@echo --PUBLISH--
python -m asv publish -o ..\build\html
@echo --END--
set %current%
