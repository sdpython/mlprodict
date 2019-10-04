@echo off
@echo --CREATE-BENCH--
python -m mlprodict asv_bench c:\temp\bench --models "LinearRegression,LogisticRegression"
@echo --RUN-BENCH--
pushd c:\temp\bench
python -m asv run --show-stderr
if exist dist\html\asv rm c:\temp\bench\html -r -f
@echo --PUBLISH--
python -m asv publish -o c:\temp\bench\html
@echo --END--
popd