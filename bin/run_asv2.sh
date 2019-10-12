echo --BENCH-CREATE--
cd _benches
python3.7 -m mlprodict asv_bench --location . --models "LinearRegression,LogisticRegression" --build "../build"
cd ..
echo --BENCH-RUN--
python3.7 -m asv run --show-stderr --config _benches/asv.conf.json
if [ -d build/html ]
then
    echo --REMOVE HTML--
    rm build/html -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config _benches/asv.conf.json -o ../build/html || exit 1