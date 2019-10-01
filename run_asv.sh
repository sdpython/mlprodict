echo --BENCH--
python3.7 -m asv run --show-stderr --config mlprodict/asv_benchmark/asv.conf.json
if [ -d dist/html/asv ]
then
    echo --REMOVE HTML--
    rm dist/html/asv -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config mlprodict/asv_benchmark/asv.conf.json -o dist/html/asv || exit 1
