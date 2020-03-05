echo --BENCH--
python -m asv run --show-stderr --config mlprodict/asv_benchmark/asv.conf.json
if [ -d dist/html/asv ]
then
    echo --REMOVE HTML--
    rm dist/html/asv -r -f
fi
if [ -d build/env ]
then
    echo --REMOVE ENV--
    rm build/env -r -f
fi
echo --PUBLISH--
python -m asv publish --config mlprodict/asv_benchmark/asv.conf.json -o ../../dist/asv/html || exit 1