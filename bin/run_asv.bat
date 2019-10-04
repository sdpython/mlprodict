@echo off
@echo --RUN-BENCH--
pushd benches
python -m asv run --show-stderr
if exist build\html rm build\html -r -f
@echo --PUBLISH--
python -m asv publish -o build\html
@echo --END--
popd
