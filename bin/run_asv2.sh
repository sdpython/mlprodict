echo --INSTALL--
python -m pip install --no-cache-dir --no-deps --index http://localhost:8067/simple/ mlprodict --extra-index-url=https://pypi.python.org/simple/
echo --BENCH-CREATE--
python -m mlprodict asv_bench --env same --location _benches --models "LinearRegression,LogisticRegression" --build "../build" || exit 1
echo --BENCH-RUN--
python -m asv run --show-stderr --config _benches/asv.conf.json --environment same --python same
if [ -d build/html ]
then
    echo --REMOVE HTML--
    rm build/html -r -f
fi
echo --PUBLISH--
python -m asv publish --config _benches/asv.conf.json -o ../build/html || exit 1