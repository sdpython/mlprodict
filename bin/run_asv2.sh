echo --BENCH-CREATE--
python3.7 -m mlprodict asv_bench --env same --location _benches --models "LinearRegression,LogisticRegression" --build "../build" || exit 1
echo --BENCH-RUN--
python3.7 -m asv run --show-stderr --config _benches/asv.conf.json --environment same --python same
if [ -d build/html ]
then
    echo --REMOVE HTML--
    rm build/html -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config _benches/asv.conf.json -o ../build/html || exit 1