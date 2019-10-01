python3.7 -m asv run --show-stderr --config mlprodict/asv_benchmark/asv.conf.json
if [ -d dist/html/asv ]
then
    rm dist/html/asv -r -f
fi
python3.7 -m asv publish --config mlprodict/asv_benchmark/asv.conf.json -o dist/html/asv || exit 1
