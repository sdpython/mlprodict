@echo off
@echo --BENCH--
python -m asv run --show-stderr --config mlprodict\asv_benchmark\asv.conf.json
if exist dist\html\asv rm dist\html\asv -r -f
@echo --PUBLISH--
python -m asv publish --config mlprodict\asv_benchmark\asv.conf.json -o ..\..\dist\html\asv