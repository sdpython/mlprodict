echo --BENCH--
cd ..
echo --RUN--
python -m asv run --show-stderr --config mlprodict/asv_benchmark/asv.conf.json
if exist dist/html/asv rmdir dist/html/asv -r -f
echo --PUBLISH--
python -m asv publish --config mlprodict/asv_benchmark/asv.conf.json -o ../../dist/asv/html || exit 1