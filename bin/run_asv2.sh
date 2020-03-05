echo --INSTALL--
python -m pip install --no-cache-dir --no-deps --upgrade --index http://localhost:8067/simple/ mlprodict --extra-index-url=https://pypi.python.org/simple/
echo --BENCHCREATE--
python -m mlprodict asv_bench --location _benches --models "LinearRegression,LogisticRegression" --build "../build" || exit 1
echo --CLEAN--
if [ -d build/html ]
then
    echo --REMOVE HTML--
    rm build/html -r -f
fi
if [ -d build/env ]
then
    echo --REMOVE ENV--
    rm build/env -r -f
fi
echo --BENCHRUN--
python -m asv run --show-stderr --config _benches/asv.conf.json -v
echo --PUBLISH--
python -m asv publish --config _benches/asv.conf.json -o build/html || exit 1